#For XML parsing
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import nltk

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#For directory listing
import os
import pprint

#For string ops
import string
import re

from gensim import corpora, models

import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn import tree, datasets, cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

pp = pprint.PrettyPrinter(indent=4)

DEBUG = True
topics_to_use = ["earn", "acquisitions", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]
topics_ids = {
	"earn" : 0,
	"acquisitions" : 1,
	"money-fx" : 2,
	"grain" : 3,
	"crude" : 4,
	"trade" : 5,
	"interest" : 6,
	"ship" : 7,
	"wheat" : 8,
	"corn" :9,
}

def debug_print(string):
	if DEBUG == True:
		print string

# DATA LOADING AND PREPROCESSING

def load_data(directory, limit=False, limit_num=1):
	xml_files = []
	#Get filenames
	for filename in os.listdir(directory):
		if filename[-3:] == 'xml':
			xml_files.append(directory + filename)

	#For each file
	articles = []
	count = 0
	limit_exceeded = False

	#Temp hack
	#xml_files = [xml_files[0], xml_files[18]]

	for xml_f in xml_files:
		if not limit_exceeded:
			debug_print("Loading " + xml_f + "...")

			# parsed_xml = ET.parse(xml_f)
			soup = BeautifulSoup(open(xml_f), ["lxml", "xml"])
			#Grab all the articles
			load_arts = soup.find_all('REUTERS')
			#print soup.prettify()
			#pp.pprint(soup.find_all('body'))
			#

			for art in load_arts:
				tmp_art = {}
				do_not_add = False
				#There should only be one text section in the article
				text_section = art.findAll('TEXT')[0]

				#pp.pprint(text_section.body)
				body_text = text_section.findAll('BODY')
				if len(body_text) > 0:
					tmp_art['body'] = body_text[0].contents[0]
				else:
					tmp_art['body'] =  ""
					do_not_add = True


				if art.attrs["LEWISSPLIT"] == "TRAIN":
					tmp_art["train"] = True
				elif art.attrs["LEWISSPLIT"] == "TEST":
					tmp_art["train"] = False
				else:
					do_not_add = True

				topics = art.findAll('TOPICS')[0].findAll('D')
				tmp_art["topics"] = []
				for i in topics:
					topic = i.contents[0]
					if topic in topics_to_use:
						tmp_art["topics"].append(topic)

				if do_not_add == False:
					if (len(tmp_art["topics"]) > 0 or tmp_art["train"] == False):
						articles.append(tmp_art)
					#pp.pprint()
				

				count += 1
				if limit == True and count >= limit_num:
					limit_exceeded = True

	return articles

def trim_and_token(articles):
	cleaned_articles = []
	#articles = articles[0:50] + articles[1500:1550]

	#Bit of a hack
	# test_count = 0
	# test_list = []
	# for i in articles:
	# 	if test_count < 50 and i["train"] == False:
	# 		test_list.append(i)
	# 		test_count += 1
	
	articles_select = articles# + test_list
	#pp.pprint(articles_select)
	for art in articles:
		#Slashes replace with a space before we tokenise
		art['body'] = re.sub("[+/<>()'\"-]", " ", art['body'])
		art['body'] = re.sub("[0-9]+th", " ", art['body'])
		tokens = nltk.word_tokenize(art['body'])
		#pp.pprint(tokens)
		punc_count = 0
		trimmed_tokens = []

		for i in range(0,len(tokens)):
			token = tokens[i]
			#Get rid of dots and commas, typically used in acronyms/nums
			token = "".join(c for c in token if c not in (",",".",))
			tokens[i] = token

			#Get rid of single punctuations
			if token in string.punctuation:
				tokens[i] = None

			if(token.lower == "reuter" or token.lower == "reuters"):
				#print "Herre"
				tokens[i] = None
			
			if token.lower in stopwords.words('english'):
				tokens[i] = None

			try:
				float(token)
				tokens[i] = None
			except ValueError as e:
				pass #do nothing


		#Get rid of the trimmed stuff & drop caps
		for token in tokens:
			if token != None:
				trimmed_tokens.append(token)

		art["body_tokens"] = trimmed_tokens

		cleaned_articles.append(art)

	return cleaned_articles

def lang_proc(articles):
	lemtz = WordNetLemmatizer()
	port = PorterStemmer()
	proc_article = []
	for article in articles:
		pos_tag_art = nltk.pos_tag(article["body_tokens"])
		ne_chunked = nltk.chunk.ne_chunk(pos_tag_art)

		token_list = []
		entity_list = {}
		full_list = []

		for token in ne_chunked:
			if isinstance(token, nltk.Tree):
				ne_list = []
				for i in token:
					ne_list.append(i)

				token_list.append({
						"token": "_".join(str(i[0]) for i in ne_list),
						"ne": True
					})
			else:
				#We remove stopwords at this stage too
				if token[0] not in stopwords.words('english'):
					tmp_lst={
						"token": token[0].lower(),
						"pos": token[1],
						"lemma": lemtz.lemmatize(token[0].lower()),
						"stem": port.stem(token[0].lower()),
						"ne": False
						}
					token_list.append(tmp_lst)



		#Now lets get the frequency of everything in the doc
		#for full_list:
		freq_tokens = {}

		for token in token_list:
			if token["ne"]:
				comb_factor = "token"
			else:
				comb_factor = "stem"

			if token[comb_factor] in freq_tokens.keys():
				#print "TRUE " + token["token"]
				freq_tokens[token[comb_factor]]["freq"] += 1
			else:
				#print "FALSE " + token["token"]
				freq_tokens[token[comb_factor]] = token
				freq_tokens[token[comb_factor]]["freq"] = 1

		article["body_token_freq"] = freq_tokens
		article["body_token_raw"] = token_list
		proc_article.append(article)
	return proc_article

# TOPIC MODEL GENERATORS

def gen_topic_model(articles):
	token_list = []
	for article in articles:
		doc = []
		for token in article['body_token_raw']:
			append_str = ""
			if token["ne"]:
				append_str = token['token']
			else:
				append_str = token['stem'] 
			if not isinstance(append_str, unicode):
				append_str = unicode(append_str, "utf-8")
			doc.append(append_str)
		token_list.append(doc)


	dictionary = corpora.Dictionary(token_list)
	corpus = [dictionary.doc2bow(token) for token in token_list]


	tfidf = models.TfidfModel(corpus)
	corpus_tdidf = tfidf[corpus]
	mod = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10)
	
	#pp.pprint(mod.show_topics(formatted=True))

	#Now run through with all our articles to get the topic weighting vector for each
	#We can classify on this
	article_groups = []
	for i in token_list:
		article_groups.append(dictionary.doc2bow(i))

	unproc_article_topic_weights = mod.inference(article_groups)
	print unproc_article_topic_weights.__class__.__name__
	for i in range(len(articles)):
		articles[i]["topic_weights"] = unproc_article_topic_weights[0][i]

	print articles[10]["topic_weights"]

	return {
		"model":mod,
		"articles": articles
		}

def gen_topic_model_ngram(articles):
	token_list = []
	for article in articles:
		doc = []
		for token in article['ngrams']:
			append_str = token
			if not isinstance(append_str, unicode):
				append_str = unicode(append_str, "utf-8")
			doc.append(append_str)
		token_list.append(doc)


	dictionary = corpora.Dictionary(token_list)
	corpus = [dictionary.doc2bow(token) for token in token_list]

	for i in corpus[0:10]:
		print i

	tfidf = models.TfidfModel(corpus)
	corpus_tdidf = tfidf[corpus]
	mod = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=10)

	#pp.pprint(mod.show_topics(formatted=True))

	#Now run through with all our articles to get the topic weighting vector for each
	#We can classify on this
	article_groups = []
	for i in token_list:
		article_groups.append(dictionary.doc2bow(i))

	unproc_article_topic_weights = mod.inference(article_groups)
	print unproc_article_topic_weights.__class__.__name__
	for i in range(len(articles)):
		articles[i]["topic_weights"] = unproc_article_topic_weights[0][i]

	print articles[10]["topic_weights"]

	return {
		"model":mod,
		"articles": articles
		}

# NGRAM GENERATOR

def gen_ngrams(articles, n):
	ngram_articles = []
	for article in articles:
		ngram_list = []
		#pp.pprint( article)
		for i in range(len(article['body_token_raw'])-n):
			ngram_list.append("_".join(article['body_token_raw'][j]["token"] for j in range(i, i+n)))
		article['ngrams'] = ngram_list
		ngram_articles.append(article)
	return ngram_articles

# DATA CONVERSION

def get_bow_classif_data(articles):
	token_train_sets = []
	token_test_sets = []
	corpus = ""
	labels = []

	for article in articles:
		#print article["train"]
		if article["train"] == True:
			doc = ""
			for i in article["body_token_raw"]:
				if i["ne"]:
					doc += " " + i["token"]
					corpus += " " + i["token"]
				else:
					doc += " " + i["stem"]
					corpus += " " + i["stem"]


			for i in article['topics']:
				token_train_sets.append(doc)
				labels.append(topics_ids[i])

		else:
			doc = ""
			for i in article["body_token_raw"]:
				
				if i["ne"]:
					doc += " " + i["token"]
				else:
					doc += " " + i["stem"]

			token_test_sets.append(doc)

	return {
		"train_tokens": token_train_sets,
		"test_tokens": token_test_sets,
		"corpus": corpus,
		"topics": labels,
	}

def get_bow_vect_data(classif_data):
	vect = TfidfVectorizer()
	vect.fit_transform([classif_data["corpus"]]).toarray()

	vect_token_sets = []

	for i in classif_data["train_tokens"]:
		vect_token_sets.append(vect.transform([i]).toarray())

	train_set = []
	for i in vect_token_sets:
		train_set.append(i[0])

	return {
		"vectorizer": vect,
		"train_vect": train_set
	}

def get_ngram_classif_data(articles):
	token_train_sets = []
	token_test_sets = []
	corpus = ""
	labels = []

	
	for article in articles:
		if article['train'] == True:
			doc = ""
			for i in article['ngrams']:
				doc += " " + i
				corpus += " " + i


			for i in article['topics']:
				token_train_sets.append(doc)
				labels.append(topics_ids[i])

		else:
			doc = ""
			for i in article['ngrams']:
				doc += " " + i

			token_test_sets.append(doc)

	return {
		"train_tokens": token_train_sets,
		"test_tokens": token_test_sets,
		"corpus": corpus,
		"topics": labels,
	}

# WORD FEATURE CLASSIFIERS

def build_run_NB(classif_data, vect_data):
	

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	# nb.fit(vect_data["train_vect"], classif_data["topics"])
	#pp.pprint(classif_data["topics"])
	# scores = cross_validation.cross_val_score(nb, vect_data["train_vect"], np.array(classif_data["topics"]), cv=10)

	np_arr_train = np.array(vect_data["train_vect"])
	np_arr_label = np.array(classif_data["topics"])

	kf = KFold(len(np_arr_label), n_folds=10, indices=True)

	fold_count = 1
	for train, test in kf:
		print "Fold " + str(fold_count)
		fold_count += 1
		nb = GaussianNB()
		#build our train & test set
		train_examples = []
		train_labels = []
		test_examples = []
		test_labels = []
		for i in train:
			train_examples.append(vect_data["train_vect"][i])
			train_labels.append(classif_data["topics"][i])
		for i in test:
			test_examples.append(vect_data["train_vect"][i])
			test_labels.append(classif_data["topics"][i])

		#train the classifier
		nb.fit(np.array(train_examples), np.array(train_labels))

		preds = nb.predict(np.array(test_examples))

		true_p = [0,0,0,0,0,0,0,0,0,0]
		false_p = [0,0,0,0,0,0,0,0,0,0]
		false_n = [0,0,0,0,0,0,0,0,0,0]
		for i in range(len(preds)):
			if preds[i] == test_labels[i]:
				true_p[test_labels[i]] += 1
			else:
				false_p[preds[i]] += 1
				false_n[test_labels[i]] += 1

		accuracy = sum(true_p) / len(preds)
		macro_prec = 0
		micro_prec = 0
		macro_reca = 0
		micro_reca = 0

		mic_p_top = 0
		mic_p_btm = 0
		mic_r_top = 0
		mic_r_btm = 0

		num_class = 10

		for i in range(num_class):
			macro_reca += true_p[i] / (true_p[i]+false_n[i])
			macro_prec += true_p[i] / (true_p[i]+false_p[i])

			mic_r_top += true_p[i]
			mic_r_btm += (true_p[i] + false_n[i])

			mic_p_top += true_p[i]
			mic_p_btm += (true_p[i] + false_p[i])


		macro_prec /= num_class
		macro_reca /= num_class

		micro_reca = mic_r_top / mic_r_btm
		micro_prec = mic_p_top / mic_p_btm

		print "Tested: " + str(len(preds))
		print "True Pos: "
		pp.pprint(true_p)
		print "False Pos: "
		pp.pprint(false_p)
		print "False Neg: "
		pp.pprint(false_n)
		print "MACRO Recall: " + str(macro_reca) + " Prec: " + str(macro_prec)
		print "MICRO Recall: " + str(micro_reca) + " Prec: " + str(micro_prec)
		print "Accuracy " + accuracy

def build_run_DecTree(classif_data, vect_data):
	dectree = tree.DecisionTreeClassifier()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	#dectree.fit(vect_data["train_vect"], classif_data["topics"])

	scores = cross_validation.cross_val_score(dectree, vect_data["train_vect"], np.array(classif_data["topics"]), cv=10)

	pp.pprint(scores)

	# #Now we vectorise each of the test entries and try to label them
	# vect_test_sets = []
	# for i in classif_data["test_tokens"]:
	# 	vect_test_sets.append(vect.transform([i]).toarray())

	# test_set = []
	# for i in vect_test_sets:
	# 	test_set.append(i[0])
	# pp.pprint(test_set[0])
	# preds = dectree.predict(test_set)
	# pp.pprint(preds)

	# count = 0
	# for i in preds[0:20]:
	# 	print "*"*45
	# 	print "\n"
	# 	print "Predicted: " + topics_to_use[i]
	# 	print classif_data["test_tokens"][count]
	# 	count += 1

	# 	print "\n"

def build_run_RF(classif_data, vect_data):
	rf = RandomForestClassifier()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	#rf.fit(vect_data["train_vect"], classif_data["topics"])

	scores = cross_validation.cross_val_score(rf, vect_data["train_vect"], np.array(classif_data["topics"]), cv=10)

	pp.pprint(scores)

	#Now we vectorise each of the test entries and try to label them
	# vect_test_sets = []
	# for i in classif_data["test_tokens"]:
	# 	vect_test_sets.append(vect.transform([i]).toarray())

	# test_set = []
	# for i in vect_test_sets:
	# 	test_set.append(i[0])
	# pp.pprint(test_set[0])
	# preds = rf.predict(test_set)
	# pp.pprint(preds)

	# count = 0
	# for i in preds[0:20]:
	# 	print "*"*45
	# 	print "\n"
	# 	print "Predicted: " + topics_to_use[i]
	# 	print classif_data["test_tokens"][count]
	# 	count += 1

	# 	print "\n"

## TOPIC MODEL CLASSIFIERS

def build_topmod_NB(articles):
	nb = GaussianNB()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	# vect = vect_data["vectorizer"]
	
	test_set = []
	train_set = []
	topic_labels = []
	for i in articles:
		if i["train"]:
			for j in i['topics']:
				train_set.append(i["topic_weights"])
				topic_labels.append(topics_ids[j])		
			else:
				test_set.append(i["topic_weights"])

	#nb.fit(train_set, topic_labels)

	scores = cross_validation.cross_val_score(nb, train_set, np.array(topic_labels), cv=10)

	pp.pprint(scores)

	#Now we vectorise each of the test entries and try to label them
	# vect_test_sets = []
	# for i in classif_data["test_tokens"]:
	# 	vect_test_sets.append(vect.transform([i]).toarray())

	# test_set = []
	# for i in vect_test_sets:
	# 	test_set.append(i[0])
	# pp.pprint(test_set[0])
	#preds = nb.predict(test_set)
	# pp.pprint(preds)


def build_topmod_DecTree(articles):
	dectree = tree.DecisionTreeClassifier()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	# vect = vect_data["vectorizer"]
	
	test_set = []
	train_set = []
	topic_labels = []
	for i in articles:
		if i["train"]:
			for j in i['topics']:
				train_set.append(i["topic_weights"])
				topic_labels.append(topics_ids[j])		
			else:
				test_set.append(i["topic_weights"])

	#dectree.fit(train_set, topic_labels)

	scores = cross_validation.cross_val_score(dectree, train_set, np.array(topic_labels), cv=10)

	pp.pprint(scores)

	#Now we vectorise each of the test entries and try to label them
	# vect_test_sets = []
	# for i in classif_data["test_tokens"]:
	# 	vect_test_sets.append(vect.transform([i]).toarray())

	# test_set = []
	# for i in vect_test_sets:
	# 	test_set.append(i[0])
	# pp.pprint(test_set[0])
	#preds = dectree.predict(test_set)
	# pp.pprint(preds)


def build_topmod_RF(articles):
	rf = RandomForestClassifier()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	# vect = vect_data["vectorizer"]
	
	test_set = []
	train_set = []
	topic_labels = []
	for i in articles:
		if i["train"]:
			for j in i['topics']:
				train_set.append(i["topic_weights"])
				topic_labels.append(topics_ids[j])		
			else:
				test_set.append(i["topic_weights"])

	#rf.fit(train_set, topic_labels)
	scores = cross_validation.cross_val_score(rf, train_set, np.array(topic_labels), cv=10)

	pp.pprint(scores)

	#Now we vectorise each of the test entries and try to label them
	# vect_test_sets = []
	# for i in classif_data["test_tokens"]:
	# 	vect_test_sets.append(vect.transform([i]).toarray())

	# test_set = []
	# for i in vect_test_sets:
	# 	test_set.append(i[0])
	# pp.pprint(test_set[0])
	#preds = rf.predict(test_set)
	# pp.pprint(preds)

# SYSTEM EXECUTORS

def run_bag_of_words(proc_arts, classif=1):
	bow_classif_data = get_bow_classif_data(proc_arts)
	bow_vect_data = get_bow_vect_data(bow_classif_data)
	
	if classif == 1:
		build_run_NB(bow_classif_data, bow_vect_data)
	elif classif == 2:
		build_run_DecTree(bow_classif_data, bow_vect_data)
	elif classif == 3:
		build_run_RF(bow_classif_data, bow_vect_data)

def run_bigram(proc_arts, classif=1):
	bigrams = gen_ngrams(proc_arts, 2)

	bigram_classif_data = get_ngram_classif_data(bigrams)
	bigram_vect_data = get_bow_vect_data(bigram_classif_data)

	if classif == 1:
		build_run_NB(bigram_classif_data, bigram_vect_data)
	elif classif == 2:
		build_run_DecTree(bigram_classif_data, bigram_vect_data)
	else:
		build_run_RF(bigram_classif_data, bigram_vect_data)

def run_trigram(proc_arts, classif=1):
	trigrams = gen_ngrams(proc_arts, 3)

	trigram_classif_data = get_ngram_classif_data(trigrams)
	trigram_vect_data = get_bow_vect_data(trigram_classif_data)

	if classif == 1:
		build_run_NB(trigram_classif_data, trigram_vect_data)
	elif classif == 2:
		build_run_DecTree(trigram_classif_data, trigram_vect_data)
	else:
		build_run_RF(trigram_classif_data, trigram_vect_data)

def run_bow_topic_model(proc_arts, classif=1):
	model = gen_topic_model(proc_arts)

	if classif == 1:
		build_topmod_NB(model["articles"])
	elif classif == 2:
		build_topmod_DecTree(model["articles"])
	elif classif == 3:
		build_topmod_RF(model["articles"])


def run_bigram_topic_model(proc_arts, classif=1):
	bigrams = gen_ngrams(proc_arts, 2)
	model = gen_topic_model_ngram(bigrams)
	
	if classif == 1:
		build_topmod_NB(model["articles"])
	elif classif == 2:
		build_topmod_DecTree(model["articles"])
	elif classif == 3:
		build_topmod_RF(model["articles"])


def run_trigram_topic_model(proc_arts, classif=1):
	trigrams = gen_ngrams(proc_arts, 3)
	model = gen_topic_model_ngram(trigrams)

	if classif == 1:
		build_topmod_NB(model["articles"])
	elif classif == 2:
		build_topmod_DecTree(model["articles"])
	elif classif == 3:
		build_topmod_RF(model["articles"])


