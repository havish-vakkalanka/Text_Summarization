from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import random
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from PyPDF2 import PdfReader
from summa.summarizer import summarize as summa_summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer

# NLTK libraries to support text processing
nltk.download('punkt')  # Tokenizer for sentences and words
nltk.download('averaged_perceptron_tagger')  # POS tagger
nltk.download('stopwords')  # Stopwords to filter out common words

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for securely signing the session cookie
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Folder where uploaded files will be stored
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}  # Allowed file extensions for uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size for uploads (16 MB)

# Initialize NLP models for summarization using HuggingFace Transformers
summarizer_t5 = pipeline("summarization", model="t5-small")
summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handle file uploads and text summarization."""
    # Check if a file is part of the request
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        text = read_pdf(file_path)
        if text:
            model_choice = request.form['model']
            summary_text = generate_summary(text, model_choice)
            session['summary'] = summary_text  # Store summary in session
            session['questions'] = generate_questions(summary_text)  # Store generated questions in session
            session['current_question'] = 0  # Initialize question index
            session['score'] = 0  # Initialize score
            return render_template('result.html', summary=summary_text)
        else:
            flash('Could not extract text from PDF')
            return redirect(url_for('home'))
    else:
        flash('Invalid file type')
        return redirect(url_for('home'))

def read_pdf(file_path):
    """Extract text from a PDF file."""
    text = ''
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ''
    return text.strip()

def generate_summary(text, model_name):
    """Generate summaries using different NLP models based on the user's choice."""
    if model_name == 'text_rank':
        return summa_summarize(text, ratio=0.1)
    elif model_name == 'lex_rank':
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, 10)
        return ' '.join([str(sentence) for sentence in summary])
    elif model_name == 'luhn':
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        summarizer = LuhnSummarizer()
        summary = summarizer(parser.document, 10)
        return ' '.join([str(sentence) for sentence in summary])
    elif model_name == 'lsa':
        parser = PlaintextParser.from_string(text, Tokenizer('english'))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 10)
        return ' '.join([str(sentence) for sentence in summary])
    elif model_name == 't5':
        summary = summarizer_t5(text, max_length=600, min_length=300, do_sample=False)
        return summary[0]['summary_text']
    elif model_name == 'bart':
        summary = summarizer_bart(text, max_length=600, min_length=300, do_sample=False)
        return summary[0]['summary_text']
    return "No valid model selected."



def generate_questions(summary):
    """Generate fill-in-the-blank questions from the summary text by identifying key nouns.
       Questions are shortened to end at the next full stop after the blank or at the sentence's end.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(summary)
    words_pos = pos_tag(words)
    nouns = [word for word, pos in words_pos if pos.startswith('NN') and word.lower() not in stop_words]

    # Shuffle the list of nouns to ensure randomness
    random.shuffle(nouns)

    questions = []
    # Limit to first five relevant nouns
    for noun in nouns[:5]:
        # Generate a sentence for each noun and replace the noun with a blank
        for sent in sent_tokenize(summary):
            if noun in word_tokenize(sent):
                sentence = sent.replace(noun, "_______", 1)
                # Truncate sentence at the first period after the blank or end the sentence if no period follows
                end_pos = sentence.find('.', sentence.find("_______")) + 1
                if end_pos > 0:
                    sentence = sentence[:end_pos]
                break
        
        correct_answer = noun
        distractors = random.sample([word for word in words if word != correct_answer and pos_tag([word])[0][1].startswith('NN')], 3)
        choices = [correct_answer] + distractors
        random.shuffle(choices)
        questions.append((sentence, choices, correct_answer))

    return questions

@app.route('/quiz')
def quiz():
    """Render the quiz page with the next question."""
    questions = session.get('questions', [])
    current_question = session.get('current_question', 0)
    if current_question < len(questions):
        question, choices, _ = questions[current_question]
        return render_template('quiz.html', question=question, choices=choices, question_number=current_question+1)
    else:
        return redirect('/score')

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """Handle quiz answer submissions and update session data accordingly."""
    selected_option = request.form['option']
    current_question = session.get('current_question', 0)
    questions = session.get('questions', [])
    _, _, correct_answer = questions[current_question]
    if selected_option == correct_answer:
        session['score'] += 1
    if current_question < len(questions) - 1:
        session['current_question'] += 1
        return redirect('/quiz')
    else:
        return redirect('/score')

@app.route('/score')
def score():
    """Display the final score after quiz completion."""
    score = session.get('score', 0)
    total_questions = len(session.get('questions', []))
    return render_template('score.html', score=score, total_questions=total_questions)

@app.route('/flashcards')
def flashcards():
    """Render the flashcards page based on the summary."""
    summary = session.get('summary', '')
    flashcards = generate_flashcards(summary)
    return render_template('flashcards.html', flashcards=flashcards)

def generate_flashcards(text, num_flashcards=5):
    """Generate flashcards from the summary text, focusing on key terms and context sentences."""
    sentences = sent_tokenize(text)
    key_terms = extract_key_terms(text)
    flashcards = []
    for term in random.sample(key_terms, min(len(key_terms), num_flashcards)):
        context_sentence = next((sentence for sentence in sentences if term in word_tokenize(sentence)), None)
        if context_sentence:
            definition = context_sentence.replace(term, "_______")
            flashcards.append((term, definition))
    return flashcards

def extract_key_terms(text):
    """Extract key terms from the text using POS tagging and filtering stopwords."""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words_pos = pos_tag(words)
    key_terms = [word for word, pos in words_pos if pos.startswith('NN') and word.lower() not in stop_words]
    return key_terms

if __name__ == '__main__':
    app.run(debug=True)
