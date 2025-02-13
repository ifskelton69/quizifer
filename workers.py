from PyPDF2 import PdfReader
from question_generation_main import QuestionGeneration
import json
import os

def pdf2text(file_path: str, file_exten: str) -> str:
    
    _content = ''
    if file_exten == 'pdf':
        with open(file_path, 'rb') as pdf_file:
            _pdf_reader = PdfReader(pdf_file)
            for p in range(len(_pdf_reader.pages)):
                _content += _pdf_reader.pages[p].extract_text()
            print('PDF operation done!')
    
    elif file_exten == 'txt':
        with open(file_path, 'r',encoding='utf-8', errors='ignore') as txt_file:
            _content = txt_file.read()
            print('TXT operation done!')
    
    return _content

def txt2questions(doc: str, n=10, o=4) -> dict:
    qGen = QuestionGeneration(n, o)
    q = qGen.generate_questions_dict(doc)

    for i in range(len(q)):
        temp = []
        for j in range(len(q[i + 1]['options'])):
            temp.append(q[i + 1]['options'][j + 1])
        q[i + 1]['options'] = temp
        
    
    return q
