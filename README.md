**PDF Summarization Tool**

**Overview**

The PDF Summarization Tool is designed to simplify the process of consuming large documents by providing concise, accurate summaries. Built using Flask, this application supports various Natural Language Processing (NLP) models like TextRank, LexRank, Luhn, LSA, T5, and BART, making it versatile for different summarization needs.

**Features**

•	**PDF Upload:** Users can upload PDF files they wish to summarize.

•	**Model Selection:** Choose from multiple NLP models to tailor the summarization.

•	**Interactive Quizzes:** Engage with the material through quizzes based on the summaries.

•	**Flashcards:** Generate flashcards for effective learning and review.

•	**Responsive Design:** Ensures a great user experience on desktops, tablets, and mobiles.

**Installation**

To set up the project on your local machine, follow these steps:

```bash
# Clone the repository
git clone https://github.com/havish-vakkalanka/pdf-summarization-tool.git

# Navigate to the project directory
cd pdf-summarization-tool

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Usage**

To run the application:

```bash
# Start the Flask server
flask run
```

Open your web browser and navigate to http://127.0.0.1:5000/ to start using the application.

**Credits**

•	[Om Sri Rohith Raj Yadav Thalla](https://github.com/rohithrajthalla) **:** Co-developer of the project.