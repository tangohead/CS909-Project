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

mode = 6
classif_mode = 1

pp = pprint.PrettyPrinter(indent=4)

articles = helper.load_data(DATA_PATH, limit=False)
clean_arts = helper.trim_and_token(articles)

proc_arts = helper.lang_proc(clean_arts)

#GENERATE CLASSIFIER DATA

# bow_classif_data = helper.get_ngram_classif_data(bigrams)
# bow_vect_data = helper.get_bow_vect_data(bow_classif_data)


#helper.build_topmod_NB(model["articles"])

if mode == 1:
	helper.run_bag_of_words(proc_arts, classif=classif_mode)
elif mode == 2:
	helper.run_bigram(proc_arts, classif=classif_mode)
elif mode == 3:
	helper.run_trigram(proc_arts, classif=classif_mode)
elif mode == 4:
	helper.run_bow_topic_model(proc_arts, classif=classif_mode)
elif mode == 5:
	helper.run_bigram_topic_model(proc_arts, classif=classif_mode)
elif mode == 6:
	helper.run_trigram_topic_model(proc_arts, classif=classif_mode)
