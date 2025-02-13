# import spacy

# nlp = spacy.load("en_core_web_sm")

# nlp.max_length = 150000000000000012

# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

import gensim.downloader as api

# Download and load the word2vec-google-news-300 model
model = api.load("word2vec-google-news-300")
