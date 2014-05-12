#  Python 2.7
# - BeautifulSoup4 (pip install BeautifulSoup4)
# - lxml (pip install lzxml)
# - nltk (pip install nltk)

#For XML parsing
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import nltk

from gensim import corpora, models
#For directory listing
import os

#Misc
import pprint

#Modules
import helper

#Config vars
DATA_PATH = "../reuters21578-xml/"

pp = pprint.PrettyPrinter(indent=4)

articles = helper.load_data(DATA_PATH, limit=False)

# pp.pprint(articles[0]['body_tokens'])
# pp.pprint(nltk.word_tokenize(articles[0]['body']))

clean_arts = helper.trim_and_token(articles)
pp.pprint(clean_arts[0]['body_tokens'])

proc_arts = helper.lang_proc(clean_arts)



bigrams = helper.gen_ngrams(proc_arts, 2)
print "ngrams"
pp.pprint(bigrams[0]['body_token_raw'])
# trigrams = helper.gen_ngrams(proc_arts, 3)

model = helper.gen_topic_model_ngram(bigrams)
#model = helper.gen_topic_model(proc_arts)

#GENERATE CLASSIFIER DATA

#bow_classif_data = helper.get_bow_classif_data(proc_arts)
# bow_vect_data = helper.get_bow_vect_data(bow_classif_data)

# bow_classif_data = helper.get_ngram_classif_data(bigrams)
# bow_vect_data = helper.get_bow_vect_data(bow_classif_data)

# # CLASSIFIERS
# helper.build_run_NB(bow_classif_data, bow_vect_data)
# helper.build_run_DecTree(bow_classif_data, bow_vect_data)
#helper.build_run_RF(bow_classif_data, bow_vect_data)

helper.build_topmod_NB(model["articles"])