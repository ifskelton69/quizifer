# import gensim.downloader as api
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords, wordnet
# import string
# import random

# class IncorrectAnswerGenerator:
#     output_file = "incorrect_answers.txt"

#     def __init__(self, document):
#         self.model = api.load("glove-wiki-gigaword-100")
#         self.all_words = set(word_tokenize(document.lower()))
#         self.stopwords = set(stopwords.words('english'))
#         self.punctuation = set(string.punctuation)

#     def get_similar_word(self, word):
#         try:
#             similar_words = self.model.most_similar(word, topn=50)
#             similar_words = [w[0] for w in similar_words if w[0] not in self.stopwords and w[0] != word and not any(char in self.punctuation for char in w[0])]
#             return random.choice(similar_words)
#         except KeyError:
#             return None

#     def get_thematic_related_word(self, word):
#         synsets = wordnet.synsets(word)
#         if not synsets:
#             return None
#         thematic_words = []
#         for synset in synsets:
#             for lemma in synset.lemmas():
#                 thematic_words.append(lemma.name().lower())
#         thematic_words = [w for w in thematic_words if w in self.all_words and w != word]
#         return random.choice(thematic_words) if thematic_words else None

#     def get_all_options_dict(self, answer, num_options):
#         options_dict = dict()
#         similar_word = self.get_similar_word(answer)
#         thematic_word = self.get_thematic_related_word(answer)

#         if similar_word:
#             options_dict[1] = similar_word
#         elif thematic_word:
#             options_dict[1] = thematic_word
#         else:
#             options_dict[1] = random.choice([word for word in self.all_words if not any(char in self.punctuation for char in word)])

#         for i in range(2, num_options + 1):
#             options_dict[i] = random.choice([word for word in (self.all_words - set(options_dict.values())) if not any(char in self.punctuation for char in word)])

#         replacement_idx = random.randint(1, num_options)
#         options_dict[replacement_idx] = answer

#         with open(self.output_file, 'a') as f:
#             f.write(f"{options_dict}\n")

#         return options_dict
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import string
import random

class IncorrectAnswerGenerator:
    output_file = "incorrect_answers.txt"

    def __init__(self, document):
        self.model = api.load("glove-wiki-gigaword-100") 
        self.all_words = set(word_tokenize(document.lower()))
        self.stopwords = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.filtered_words = [word for word in self.all_words if word not in self.stopwords and word not in self.punctuation]

    def get_similar_word(self, word):
        try:
            similar_words = self.model.most_similar(word, topn=50)
            similar_words = [w[0] for w in similar_words if w[0] in self.filtered_words and w[0] != word]
            return random.choice(similar_words) if similar_words else None
        except KeyError:
            return None

    def get_thematic_related_word(self, word):
        synsets = wordnet.synsets(word)
        if not synsets:
            return None
        thematic_words = []
        for synset in synsets:
            for lemma in synset.lemmas():
                thematic_words.append(lemma.name().lower())
        thematic_words = [w for w in thematic_words if w in self.filtered_words and w != word]
        return random.choice(thematic_words) if thematic_words else None

    def get_all_options_dict(self, answer, num_options):
        options_dict = dict()

        similar_word = self.get_similar_word(answer)
        thematic_word = self.get_thematic_related_word(answer)

        if similar_word:
            options_dict[1] = similar_word
        elif thematic_word:
            options_dict[1] = thematic_word
        else:
            options_dict[1] = random.choice(self.filtered_words)

        for i in range(2, num_options + 1):
            options_dict[i] = random.choice([word for word in self.filtered_words if word not in options_dict.values()])

        replacement_idx = random.randint(1, num_options)
        options_dict[replacement_idx] = answer

        with open(self.output_file, 'a') as f:
            f.write(f"{options_dict}\n")

        return options_dict
