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

quizifer:
  __pycache__:
    - incorrect_answer_generation.cpython-3xx.pyc
    - question_extraction.cpython-3xx.pyc
    - question_generation_main.cpython-3xx.pyc
    - workers.cpython-3xx.pyc
  env:
    Lib/
      site-packages/
    Scripts/
    pyvenv.cfg
  pdf:
    - 0124114547cloud.pdf
  assets:
    css/
    scripts/
  templates/
  app.py
  downloadwordnet.py
  incorrect_answer_generation.py
  incorrect_answers.txt
  package.json
  package-lock.json
  question_extraction.py
  question_extraction_output.txt
  question_generation_main.py
  qgenDummy.py
  README.md


```bash
git clone https://github.com/your-username/quizifer.git
cd quizifer

# Create and activate virtual environment
python -m venv env
source env/bin/activate       # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt


pip install flask nltk pymupdf
