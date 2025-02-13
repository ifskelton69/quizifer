import nltk
import spacy
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import random

class QuestionExtractor:
    def __init__(self, num_questions):
        self.num_questions = num_questions
        self.stop_words = set(stopwords.words('english'))

        self.ner_tagger = spacy.load('en_core_web_md')

        self.vectorizer = TfidfVectorizer()
        self.questions_dict = dict()
        self.random_variable = random.random()

    def get_questions_dict(self, document):
        self.candidate_keywords = self.get_candidate_entities(document)
        self.set_tfidf_scores(document)
        self.rank_keywords()
        self.form_questions()
        return self.questions_dict

    def get_filtered_sentences(self, document):
        sentences = sent_tokenize(document)

        return [self.filter_sentence(sentence) for sentence in sentences]

    def filter_sentence(self, sentence):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        nouns = [word for word, pos in tagged_words if pos.startswith('N')]
        return ' '.join(nouns)

    def get_candidate_entities(self, document):
        entities = self.ner_tagger(document)
        entity_list = []

        for ent in entities.ents:
            entity_list.append(ent.text)

        return list(set(entity_list))

    def set_tfidf_scores(self, document):
        self.unfiltered_sentences = sent_tokenize(document)
        self.filtered_sentences = self.get_filtered_sentences(document)

        self.word_score = dict()
        self.sentence_for_max_word_score = dict()

        tf_idf_vector = self.vectorizer.fit_transform(self.filtered_sentences)
        feature_names = self.vectorizer.get_feature_names_out()
        tf_idf_matrix = tf_idf_vector.todense().tolist()

        num_sentences = len(self.unfiltered_sentences)
        num_features = len(feature_names)

        for i in range(num_features):
            word = feature_names[i]
            self.sentence_for_max_word_score.setdefault(word, []).append("")

            tot = 0.0
            cur_max = 0.0

            for j in range(num_sentences):
                tot += tf_idf_matrix[j][i]

                if tf_idf_matrix[j][i] > cur_max:
                    cur_max = tf_idf_matrix[j][i]
                    self.sentence_for_max_word_score[word][-1] = self.unfiltered_sentences[j]

            # self.word_score[word] = tot / num_sentences
            self.word_score[word] = tot / num_sentences + (random.random() - 0.50002983929) * 0.111


            try:
                print(f"Word: {word}, Score: {self.word_score[word]}, Max Sentence: {self.sentence_for_max_word_score[word][-1]}")
            except Exception as e:
                print(f"Exception during debugging print: {e}")
        print("Final word_score:", self.word_score)
        print("Final sentence_for_max_word_score:", self.sentence_for_max_word_score)

    def get_keyword_score(self, keyword):
        score = 0.0
        for word in word_tokenize(keyword):
            if word in self.word_score:
                score += self.word_score[word]
        return score + self.random_variable


    def get_corresponding_sentence_for_keyword(self, keyword):
        words = word_tokenize(keyword)
        for word in words:
            if word not in self.sentence_for_max_word_score:
                continue

            sentences = self.sentence_for_max_word_score[word]

            for sentence in sentences:
                all_present = all(w in sentence for w in words)

                if all_present:
                    return sentence

        return ""

    def rank_keywords(self):
        self.candidate_triples = []

        for candidate_keyword in self.candidate_keywords:
            self.candidate_triples.append([
                self.get_keyword_score(candidate_keyword),
                candidate_keyword,
                self.get_corresponding_sentence_for_keyword(candidate_keyword)
            ])

        self.candidate_triples.sort(reverse=True)

    def form_questions(self):
        used_sentences = list()
        idx = 0
        cntr = 1
        num_candidates = len(self.candidate_triples)

        while cntr <= self.num_questions and idx < num_candidates:
            candidate_triple = self.candidate_triples[idx]

            if candidate_triple[2] not in used_sentences:
                used_sentences.append(candidate_triple[2])

                if candidate_triple[1] in candidate_triple[2]:
                    masked_question = candidate_triple[2].replace(
                        candidate_triple[1],
                        '_' * len(candidate_triple[1])
                    )
                else:
                    idx += 1
                    continue

                question_type = random.choice(["definition", "identification", "elaboration", "comparison", "example"])

                if question_type == "definition":
                    modified_question = f"What does the author refer to as {masked_question}?"
                elif question_type == "identification":
                    modified_question = f"What is {masked_question}?"
                else:
                    modified_question = f"{masked_question}?"

                self.questions_dict[cntr] = {
                    "question": modified_question,
                    "answer": candidate_triple[1]
                }

                cntr += 1
            idx += 1
