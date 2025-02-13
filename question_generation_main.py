from question_extraction import QuestionExtractor
from incorrect_answer_generation import IncorrectAnswerGenerator
import re
from nltk import sent_tokenize
import random
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import random

class QuestionGeneration:
    def __init__(self, num_questions, num_options):
        self.num_questions = num_questions
        self.num_options = num_options
        self.question_extractor = QuestionExtractor(self.num_questions)

    @staticmethod
    def clean_sentence(sentence):
        cleaned_sentence = re.sub(r'([^\s\w]|_)+', '', sentence)
        cleaned_sentence = re.sub(' +', ' ', cleaned_sentence)
        return cleaned_sentence.strip() + '.'

    def clean_text(self, text):
        text = text.replace('\n', ' ')
        sentences = sent_tokenize(text)
        cleaned_sentences = [self.clean_sentence(sentence) for sentence in sentences]
        return ' '.join(cleaned_sentences)

    def generate_questions_dict(self, document):
        random.seed()

        cleaned_document = self.clean_text(document)
        self.questions_dict = self.question_extractor.get_questions_dict(cleaned_document)
        self.incorrect_answer_generator = IncorrectAnswerGenerator(cleaned_document)

        for i in range(1, self.num_questions + 1):
            if i not in self.questions_dict:
                continue
            answer = self.questions_dict[i]["answer"]
            self.questions_dict[i]["options"] = self.incorrect_answer_generator.get_all_options_dict(
                answer, self.num_options
            )

        return self.questions_dict