# question_generation_main.py

from question_extraction import QuestionExtractor
from incorrect_answer_generation import IncorrectAnswerGenerator
import re
from nltk import sent_tokenize
import random

class QuestionGeneration:
    def __init__(self, num_questions, num_options):
        # self.num_questions = num_questions + random.randint(0, 20)
        self.num_questions = num_questions
        self.num_options = num_options
        self.question_extractor = QuestionExtractor(self.num_questions)

    def clean_text(self, text):
        text = text.replace('\n', ' ')
        sentences = sent_tokenize(text)
        cleaned_text = ""

        for sentence in sentences:
            cleaned_sentence = re.sub(r'([^\s\w]|_)+', '', sentence)
            cleaned_sentence = re.sub(' +', ' ', cleaned_sentence)
            cleaned_text += cleaned_sentence

            if cleaned_text[-1] == ' ':
                cleaned_text = cleaned_text.strip()
                cleaned_text += '.'
            else:
                cleaned_text += '.'
            cleaned_text += ' '

        return cleaned_text

    def generate_questions_dict(self, document):
        document = self.clean_text(document)
        self.questions_dict = self.question_extractor.get_questions_dict(document)
        self.incorrect_answer_generator = IncorrectAnswerGenerator(document)

        for i in range(1, self.num_questions + 1):
            if i not in self.questions_dict:
                continue
            self.questions_dict[i]["options"] = self.incorrect_answer_generator.get_all_options_dict(
                self.questions_dict[i]["answer"], self.num_options
            )

        return self.questions_dict
