#  Python 2.7
# - BeautifulSoup4 (pip install BeautifulSoup4)
# - lxml (pip install lzxml)
# - nltk (pip install nltk)

#For XML parsing
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import nltk

import argparse

from gensim import corpora, models
#For directory listing
import os

#Misc
import pprint

#Modules
import helper

#Config vars
DATA_PATH = "../reuters21578-xml/"

parser = argparse.ArgumentParser(description="Classifier!")
parser.add_argument("mode", metavar="M", type=int, help="the id of the feature mode to use")
parser.add_argument("classifier", metavar="C", type=int, help="the id of the classifier to use")
parser.add_argument("limit", metavar="L", type=int, help="1 to limit, 0 otherwise")
parser.add_argument("limit_size", metavar="S", type=int, help="size of limit (use 0 if no limit)")

args = parser.parse_args()

mode = args.mode
classif_mode = args.classifier
limit_load = False
if args.limit == 1:
	limit_load = True
limit_size = args.limit_size

pp = pprint.PrettyPrinter(indent=4)


# Load then preprocess
articles = helper.load_data(DATA_PATH, limit=limit_load, limit_num=limit_size)
clean_arts = helper.trim_and_token(articles)

proc_arts = helper.lang_proc(clean_arts)

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


helper.log_close()