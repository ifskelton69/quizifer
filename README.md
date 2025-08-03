# 📘 Quizifer

**Quizifer** is an intelligent quiz generator that automatically extracts text from PDF files and generates quizzes based on the content. It lets you customize the number of questions and uses NLP techniques to create meaningful MCQs, including both correct and incorrect answers.

---

## 🚀 Features

- 📄 Upload any PDF document
- 🧠 Automatically generates quiz questions using NLP
- ❓ Each question includes one correct and three incorrect options
- ⚙️ Customize the number of questions to generate
- 📦 Organized project structure for easy extension
- 🌐 Flask-based web interface (optional)

---

## 🛠️ Tech Stack

- **Python**
- **NLTK** – Natural Language Toolkit
- **WordNet** – For generating incorrect options
- **PyMuPDF / fitz** – For PDF text extraction
- **Flask** – (for the web app interface)

---

## 📁 Project Structure



---

## 📦 Installation

quizifer/
├── pycache/ # Compiled Python files
│
├── env/ # Virtual environment
│ ├── Lib/site-packages/
│ ├── Scripts/
│ └── pyvenv.cfg
│
├── pdf/ # PDF upload folder
│ └── 0124114547cloud.pdf
│
├── assets/ # Frontend assets
│ ├── css/
│ └── scripts/
│
├── templates/ # HTML templates for Flask app
│
├── app.py # Main app file (Flask or CLI)
├── question_extraction.py # Extracts sentences from PDF text
├── question_generation_main.py # Generates question-answer pairs
├── incorrect_answer_generation.py # Creates distractor options
├── qgenDummy.py # Optional mock or test script
├── downloadwordnet.py # Downloads NLTK WordNet
├── incorrect_answers.txt # Stores incorrect options
├── question_extraction_output.txt # Output log for questions
├── README.md # This file
├── package.json # For future frontend (optional)
└── package-lock.json

yaml
Copy
Edit


```bash
git clone https://github.com/your-username/quizifer.git
cd quizifer

# Create and activate virtual environment
python -m venv env
source env/bin/activate       # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt


pip install flask nltk pymupdf
