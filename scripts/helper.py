# Defaults to float division
from __future__ import division

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
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn import tree, datasets, cross_validation, metrics, mixture
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances

pp = pprint.PrettyPrinter(indent=4)

bigram_stash = "bigram_freq_stash.p"
trigram_stash = "trigram_freq_stash.p"

DEBUG = True
topics_to_use = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]
topics_ids = {
	"earn" : 0,
	"acq" : 1,
	"money-fx" : 2,
	"grain" : 3,
	"crude" : 4,
	"trade" : 5,
	"interest" : 6,
	"ship" : 7,
	"wheat" : 8,
	"corn" :9,
}

log = open("output.txt", "w")

def debug_print(string):
	if DEBUG == True:
		print string

def close_log():
	log.close()

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
	print limit_num

	#Temp hack
	#xml_files = [xml_files[0], xml_files[18]]

	for xml_f in xml_files:
		if limit_exceeded == False:
			log.write("Loading " + xml_f + "..." + "\n")
			log.flush()

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
					topic = str(i.contents[0])
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

	count = 0
	for art in articles:
		if count % 500 == 0:
			log.write("Tokenising article " + str(count) + "\n")
			log.flush()
		count += 1
		#Slashes replace with a space before we tokenise
		art['body'] = re.sub("[+/<>()'\"-]", " ", art['body'])
		art['body'] = re.sub("[0-9]+th", " ", art['body'])
		tokens = nltk.word_tokenize(art['body'])
		#pp.pprint(tokens)
		punc_count = 0
		trimmed_tokens = []

		for i in range(0,len(tokens)):
			token = tokens[i]
			#Get rid of dots and commas, typically used in acronyms
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
	count = 0
	for article in articles:
		if count % 500 == 0:
			log.write("Processing article " + str(count) + "\n")
			log.flush()
		count += 1

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
						#"pos": token[1],
						"lemma": lemtz.lemmatize(token[0].lower()),
						"stem": port.stem(token[0].lower()),
						"ne": False
						}
					token_list.append(tmp_lst)

		new_art = {
			"train": article["train"],
			"body_token_raw": token_list,
			"topics": article["topics"],
		}
		#article["body_token_raw"] = token_list
		proc_article.append(new_art)
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
	#print unproc_article_topic_weights.__class__.__name__
	for i in range(len(articles)):
		articles[i]["topic_weights"] = unproc_article_topic_weights[0][i]

	#print articles[10]["topic_weights"]

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
	#print unproc_article_topic_weights.__class__.__name__
	for i in range(len(articles)):
		articles[i]["topic_weights"] = unproc_article_topic_weights[0][i]

	#print articles[10]["topic_weights"]

	return {
		"model":mod,
		"articles": articles
		}

# NGRAM GENERATOR

def gen_ngrams(articles, n):
	ngram_articles = []

	new_ngram_article_list = []
	count = 0

	if n == 2:
		tok_stash = "trimmed_big.p"
	else:
		tok_stash = "trimmed_trig.p"

	if os.path.isfile(tok_stash) == False:
		for article in articles:
			if count % 500 == 0:
				log.write("Generating ngrams, " + str(count) + " articles done\n")
				log.flush()
			count += 1

			ngram_list = []
			#pp.pprint( article)
			for i in range(len(article['body_token_raw'])-n):
				ngram_list.append("_".join(article['body_token_raw'][j]["token"] for j in range(i, i+n)))
			
			ngram_articles.append({
				"ngrams": ngram_list,
				"topics": article["topics"],
				"train": article["train"]
			})
		ngram_freq_dict = {}
		if n == 2:
			stash = bigram_stash
		else:
			stash = trigram_stash

		if os.path.isfile(stash) == False:
			count = 0
			for article in ngram_articles:
				if count % 500 == 0:
					log.write("Counting ngrams, " + str(count) + " articles done\n")
					log.flush()
				count += 1
				for ngram in article["ngrams"]:
					if ngram in ngram_freq_dict.keys():
						ngram_freq_dict[ngram] += 1
					else:
						ngram_freq_dict[ngram] = 1
			store = open(stash, "wb")
			pickle.dump(ngram_freq_dict, store)
			store.close()
		else:
			store = open(stash, "rb")
			ngram_freq_dict = pickle.load(store)
			store.close()

		log.write("Getting Freqs\n")
		ngrams_to_keep = []

		if n == 2:
			stash = "bigram_keep.p"
		else:
			stash = "trigram_keep.p"

		if os.path.isfile(stash) == False:
			for i in ngram_freq_dict:
				if ngram_freq_dict[i] >= 10:
					ngrams_to_keep.append(i)

			store = open(stash, "wb")
			pickle.dump(ngrams_to_keep, store)
			store.close()
		else:
			store = open(stash, "rb")
			ngrams_to_keep = pickle.load(store)
			store.close()

		for i in ngram_articles:
			new_ngrams = []
			for ngram in i["ngrams"]:
				if ngram in ngrams_to_keep:
					new_ngrams.append(ngram)
			
			if len(new_ngrams) > 0:
				i["ngrams"] = new_ngrams
				new_ngram_article_list.append(i)

		ngram_articles = None

		store = open(tok_stash, "wb")
		pickle.dump(new_ngram_article_list, store)
		store.close()
	else:
		store = open(tok_stash, "rb")
		new_ngram_article_list = pickle.load(store)
		store.close()

	return new_ngram_article_list

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
	vect.fit_transform([classif_data["corpus"]])

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

def cluster_kmeans(classif_data, vect_data):
	km = KMeans(n_clusters=10)

	np_arr_train = np.array(vect_data["train_vect"])
	np_arr_label = np.array(classif_data["topics"])
	np_arr_test = np.array(vect_data["test_vect"])

	km.fit(np_arr_train)
	#pp.pprint(km.labels_)
	print "Kmeans"
	print "Silhouette" 
	sil_score = 0
	for i in range(10):
		sil_score += metrics.silhouette_score(np_arr_train, km.labels_, metric='euclidean', sample_size =5000)

	print sil_score/10

	print "Homog Comp VMeasure"
	pp.pprint(metrics.homogeneity_completeness_v_measure(np_arr_label, km.labels_))

	return km.labels_

def cluster_EM(classif_data, vect_data):
	gmm = mixture.GMM(n_components=10)

	np_arr_train = np.array(vect_data["train_vect"])
	np_arr_label = np.array(classif_data["topics"])
	np_arr_test = np.array(vect_data["test_vect"])

	print "GMM"

	gmm.fit(np_arr_train)
	print "Silhouette"
	sil_score = 0
	for i in range(10):
		sil_score += metrics.silhouette_score(np_arr_train, gmm.predict(np_arr_train), metric='euclidean', sample_size=5000)

	print sil_score/10

	print "Homog Comp VMeasure"
	pp.pprint(metrics.homogeneity_completeness_v_measure(np_arr_label, gmm.predict(np_arr_train)))

	return gmm.predict(np_arr_train)

def cluster_DB(classif_data, vect_data):
	db = DBSCAN()

	np_arr_train = np.array(vect_data["train_vect"])
	np_arr_label = np.array(classif_data["topics"])
	np_arr_test = np.array(vect_data["test_vect"])

	print "DB"

	db.fit(np_arr_train)
	#pp.pprint(db.labels_)
	#pp.pprint(metrics.silhouette_score(np_arr_train, db.labels_, metric='euclidean'))

	print "Silhouette" 
	sil_score = 0
	for i in range(10):
		sil_score += metrics.silhouette_score(np_arr_train, db.labels_, metric='euclidean', sample_size=5000)

	print sil_score/10

	print "Homog Comp VMeasure"
	pp.pprint(metrics.homogeneity_completeness_v_measure(np_arr_label, db.labels_))

	return db.labels_


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
def run_k_fold(classif, train_set, label_set, filename):

	kf = KFold(len(label_set), n_folds=10, indices=True)

	fold_count = 1
	print_str = ""
	for train, test in kf:
		log.write("Fold " + str(fold_count) + "\n")
		log.flush()
		fold_count += 1
		cf = classif()
		#build our train & test set
		train_examples = []
		train_labels = []
		test_examples = []
		test_labels = []
		for i in train:
			train_examples.append(train_set[i])
			train_labels.append(label_set[i])
		for i in test:
			test_examples.append(train_set[i])
			test_labels.append(label_set[i])

		#train the classifier
		cf.fit(np.array(train_examples), np.array(train_labels))

		preds = cf.predict(np.array(test_examples))

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

		macro_prec = 0.0
		micro_prec = 0.0
		macro_reca = 0.0
		micro_reca = 0.0

		mic_p_top = 0.0
		mic_p_btm = 0.0
		mic_r_top = 0.0
		mic_r_btm = 0.0

		num_class = 10

		for i in range(num_class):
			if (true_p[i]+false_n[i]) > 0.0:
				macro_reca += true_p[i] / (true_p[i]+false_n[i])
			if (true_p[i] + false_p[i]) > 0.0:
				macro_prec += true_p[i] / (true_p[i]+false_p[i])

			mic_r_top += true_p[i]
			mic_r_btm += (true_p[i] + false_n[i])

			mic_p_top += true_p[i]
			mic_p_btm += (true_p[i] + false_p[i])


		macro_prec = macro_prec / num_class
		macro_reca = macro_reca / num_class

		micro_reca = mic_r_top / mic_r_btm
		micro_prec = mic_p_top / mic_p_btm

		print_str += "*~"*40 + "\n"
		print_str += "Tested: " + str(len(preds)) + "\n"
		print_str += "True Pos: " + "\n"
		print_str += str(true_p) + "\n"
		print_str += "False Pos: " + "\n"
		print_str += str(false_p)  + "\n"
		print_str += "False Neg: " + "\n"
		print_str += str(false_n)  + "\n"

		print_str += "MACRO Recall: " + str(macro_reca) + " Prec: " + str(macro_prec)  + "\n"
		print_str += "MICRO Recall: " + str(micro_reca) + " Prec: " + str(micro_prec) + "\n"
		print_str += "Accuracy " + str(accuracy) + "\n"

	f = open(filename, "w")
	f.write(print_str)
	f.close()

def build_run_NB(classif_data, vect_data, filename):
	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	np_arr_train = vect_data["train_vect"]
	np_arr_label = classif_data["topics"]

	run_k_fold(GaussianNB, np_arr_train, np_arr_label, filename)

def build_run_DecTree(classif_data, vect_data, filename):
	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	np_arr_train = vect_data["train_vect"]
	np_arr_label = classif_data["topics"]

	run_k_fold(tree.DecisionTreeClassifier, np_arr_train, np_arr_label, filename)

def build_run_RF(classif_data, vect_data, filename):
	rf = RandomForestClassifier()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	np_arr_train = vect_data["train_vect"]
	np_arr_label = classif_data["topics"]

	run_k_fold(RandomForestClassifier, np_arr_train, np_arr_label, filename)


## TOPIC MODEL CLASSIFIERS

def build_topmod_NB(articles, filename):
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

	#scores = cross_validation.cross_val_score(nb, train_set, np.array(topic_labels), cv=10)

	#pp.pprint(scores)

	run_k_fold(GaussianNB, np.array(train_set), np.array(topic_labels), filename)

def build_topmod_DecTree(articles,  filename):
	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	
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

	run_k_fold(tree.DecisionTreeClassifier, np.array(train_set), np.array(topic_labels), filename)

def build_topmod_RF(articles, filename):
	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	
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

	run_k_fold(RandomForestClassifier, np.array(train_set), np.array(topic_labels), filename)

# SYSTEM EXECUTORS

def run_bag_of_words(proc_arts, classif=1):
	bow_classif_data = get_bow_classif_data(proc_arts)
	bow_vect_data = get_bow_vect_data(bow_classif_data)
	
	if classif == 1:
		build_run_NB(bow_classif_data, bow_vect_data, "bow-nb")
	elif classif == 2:
		build_run_DecTree(bow_classif_data, bow_vect_data, "bow-tree")
	elif classif == 3:
		build_run_RF(bow_classif_data, bow_vect_data, "bow-rf")

def run_bigram(proc_arts, classif=1):
	bigrams = gen_ngrams(proc_arts, 2)

	proc_arts = None

	bigram_classif_data = get_ngram_classif_data(bigrams)
	bigrams = None
	bigram_vect_data = get_bow_vect_data(bigram_classif_data)

	

	if classif == 1:
		build_run_NB(bigram_classif_data, bigram_vect_data, "big-nb")
	elif classif == 2:
		build_run_DecTree(bigram_classif_data, bigram_vect_data, "big-tree")
	else:
		build_run_RF(bigram_classif_data, bigram_vect_data, "big-rf")

def run_trigram(proc_arts, classif=1):
	trigrams = gen_ngrams(proc_arts, 3)

	proc_arts = None

	trigram_classif_data = get_ngram_classif_data(trigrams)
	trigrams = None
	trigram_vect_data = get_bow_vect_data(trigram_classif_data)

	

	if classif == 1:
		build_run_NB(trigram_classif_data, trigram_vect_data, "trig-nb")
	elif classif == 2:
		build_run_DecTree(trigram_classif_data, trigram_vect_data, "trig-tree")
	else:
		build_run_RF(trigram_classif_data, trigram_vect_data, "trig-rf")

def run_bow_topic_model(proc_arts, classif=1):
	model = gen_topic_model(proc_arts)

	if classif == 1:
		build_topmod_NB(model["articles"], "bow-topmod-nb")
	elif classif == 2:
		build_topmod_DecTree(model["articles"], "bow-topmod-tree")
	elif classif == 3:
		build_topmod_RF(model["articles"], "bow-topmod-rf")

def run_bigram_topic_model(proc_arts, classif=1):
	bigrams = gen_ngrams(proc_arts, 2)
	model = gen_topic_model_ngram(bigrams)
	
	if classif == 1:
		build_topmod_NB(model["articles"], "big-topmod-nb")
	elif classif == 2:
		build_topmod_DecTree(model["articles"], "big-topmod-tree")
	elif classif == 3:
		build_topmod_RF(model["articles"], "big-topmod-rf")

def run_trigram_topic_model(proc_arts, classif=1):
	trigrams = gen_ngrams(proc_arts, 3)
	model = gen_topic_model_ngram(trigrams)

	if classif == 1:
		build_topmod_NB(model["articles"], "trig-topmod-nb")
	elif classif == 2:
		build_topmod_DecTree(model["articles"], "trig-topmod-tree")
	elif classif == 3:
		build_topmod_RF(model["articles"], "trig-topmod-rf")


def run_bag_of_words_cluster(proc_arts, classif=1):
	bow_classif_data = get_bow_classif_data_test(proc_arts)
	bow_vect_data = get_bow_vect_data_test(bow_classif_data)
	
	# if classif == 1:
	test_preds = clust_DecTree(bow_classif_data, bow_vect_data)
	k_preds = cluster_kmeans(bow_classif_data, bow_vect_data)
	db_preds = cluster_DB(bow_classif_data, bow_vect_data)
	em_preds = cluster_EM(bow_classif_data, bow_vect_data)


	correct = {"em": 0, "k": 0, "db": 0}
	for i in range(len(classif_data["topics"])):
		if classif_data["topics"][i] == k_preds[i]:
			correct["k"] += 1	
		if classif_data["topics"][i] == em_preds[i]:
			correct["em"] += 1	
		if classif_data["topics"][i] == db_preds[i]:
			correct["db"] += 1		

	print "Comparisons \n"
	print "Actual: " + str(len(preds))
	print "K-means: " + str(correct["k"])
	print "DBSCAN: " + str(correct["db"])
	print "EM: " + str(correct["em"])
	# elif classif == 2:
	# 	build_run_DecTree(bow_classif_data, bow_vect_data, "bow-tree")
	# elif classif == 3:
	# 	build_run_RF(bow_classif_data, bow_vect_data, "bow-rf")


def run_bag_of_words_test(proc_arts):
	bow_classif_data = get_bow_classif_data_test(proc_arts)
	bow_vect_data = get_bow_vect_data_test(bow_classif_data)

	test_DecTree(bow_classif_data, bow_vect_data, "bow-tree-test")


def test_DecTree(classif_data, vect_data, filename):
	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	dt = tree.DecisionTreeClassifier()
	vect = vect_data["vectorizer"]

	np_arr_train = np.array(vect_data["train_vect"])
	np_arr_test = np.array(vect_data["test_vect"])
	np_arr_label = np.array(classif_data["topics"])
	test_labels = classif_data["test_topics"]

	#train the classifier
	dt.fit(np_arr_train, np_arr_label)

	preds = dt.predict(np_arr_test)

	print_str = ""

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
	num_class = 10

	prec = sum(true_p) / (sum(true_p) + sum(false_p))
	reca = sum(true_p) / (sum(true_p) + sum(false_n)) 


	print_str += "*~"*40 + "\n"
	print_str += "Tested: " + str(len(preds)) + "\n"
	print_str += "True Pos: " + "\n"
	print_str += str(true_p) + "\n"
	print_str += "False Pos: " + "\n"
	print_str += str(false_p)  + "\n"
	print_str += "False Neg: " + "\n"
	print_str += str(false_n)  + "\n"

	print_str += "Recall: " + str(reca) + " Prec: " + str(prec)  + "\n"
	print_str += "Accuracy " + str(accuracy) + "\n"

	f = open(filename, "w")
	f.write(print_str)
	f.close()

def clust_DecTree(classif_data, vect_data):
	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	dt = tree.DecisionTreeClassifier()
	vect = vect_data["vectorizer"]

	np_arr_train = np.array(vect_data["train_vect"])
	np_arr_test = np.array(vect_data["test_vect"])
	np_arr_label = np.array(classif_data["topics"])
	test_labels = classif_data["test_topics"]

	#train the classifier
	dt.fit(np_arr_train, np_arr_label)

	preds = dt.predict(np_arr_train)

	return preds

def get_bow_vect_data_test(classif_data):
	vect = TfidfVectorizer()
	vect.fit_transform([classif_data["corpus"]])

	#Before we begin, get rid of any test articles with no topic
	vect_token_sets = []
	vect_test_sets = []
	for i in classif_data["train_tokens"]:
		vect_token_sets.append(vect.transform([i]).toarray())

	for i in classif_data["test_tokens"]:
		vect_test_sets.append(vect.transform([i]).toarray())


	train_set = []
	test_set = []
	for i in vect_token_sets:
		train_set.append(i[0])
	for i in vect_test_sets:
		test_set.append(i[0])

	return {
		"vectorizer": vect,
		"train_vect": train_set,
		"test_vect": test_set
	}

def get_bow_classif_data_test(articles):
	token_train_sets = []
	token_test_sets = []
	corpus = ""
	labels = []
	test_labels = []
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

			for i in article["topics"]:
				token_test_sets.append(doc)
				test_labels.append(topics_ids[i])

	return {
		"train_tokens": token_train_sets,
		"test_tokens": token_test_sets,
		"corpus": corpus,
		"topics": labels,
		"test_topics": test_labels
	}
