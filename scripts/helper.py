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

from sklearn.naive_bayes import GaussianNB
from sklearn import tree, datasets
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
	xml_files = [xml_files[0], xml_files[18]]

	for xml_f in xml_files:
		if not limit_exceeded:
			debug_print("Loading " + xml_f + "...")

			# parsed_xml = ET.parse(xml_f)
			soup = BeautifulSoup(open(xml_f))
			#Grab all the articles
			load_arts = soup.find_all('reuters')

			for art in load_arts:
				tmp_art = {}
				do_not_add = False
				#There should only be one text section in the article
				text_section = art.findAll('text')[0]

				# if text_section.dateline != None:
				# 	tmp_art['dateline'] = text_section.dateline.string
				# else:
				# 	tmp_art['dateline'] = ""

				if text_section.body != None:
					tmp_art['body'] = text_section.body.string
				else:
					tmp_art['body'] =  ""


				if art.attrs["lewissplit"] == "TRAIN":
					tmp_art["train"] = True
				elif art.attrs["lewissplit"] == "TEST":
					tmp_art['train'] = False
				else:
					do_not_add = True

				topics = art.findAll('topics')[0].findAll('d')
				#print topics
				tmp_art["topics"] = []
				for i in topics:
					topic = i.contents[0]
					if topic in topics_to_use:
						tmp_art["topics"].append(topic)

				if len(tmp_art["topics"]) > 0 and do_not_add == False:
					articles.append(tmp_art)
					pp.pprint(tmp_art)

				count += 1
				if limit == True and count >= limit_num:
					limit_exceeded = True

	return articles

def trim_and_token(articles):
	cleaned_articles = []
	#articles = articles[0:50] + articles[1500:1550]

	#Bit of a hack
	test_count = 0
	test_list = []
	for i in articles:
		if test_count < 50 and i["train"] == False:
			test_list.append(i)
			test_count += 1

	articles = articles + test_list

	for art in articles[0:150]:
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

			if(token.lower in ["reuter", "reuters"]):
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

		art['body_tokens'] = trimmed_tokens

		cleaned_articles.append(art)

	return cleaned_articles

def lang_proc(articles):
	lemtz = WordNetLemmatizer()
	port = PorterStemmer()
	proc_article = []
	for article in articles:
		pos_tag_art = nltk.pos_tag(article['body_tokens'])
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

		article['body_token_freq'] = freq_tokens
		article['body_token_raw'] = token_list
		proc_article.append(article)
	return proc_article

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

def get_bow_classif_data(articles):
	token_train_sets = []
	token_test_sets = []
	corpus = ""
	labels = []

	for article in articles:
		if article['train'] == True:
			doc = ""
			for i in article['body_token_raw']:
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
			for i in article['body_token_raw']:
				
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




def build_run_NB(classif_data, vect_data):
	nb = GaussianNB()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	nb.fit(vect_data["train_vect"], classif_data["topics"])

	#Now we vectorise each of the test entries and try to label them
	vect_test_sets = []
	for i in classif_data["test_tokens"]:
		vect_test_sets.append(vect.transform([i]).toarray())

	test_set = []
	for i in vect_test_sets:
		test_set.append(i[0])
	pp.pprint(test_set[0])
	preds = nb.predict(test_set)
	pp.pprint(preds)

	count = 0
	for i in preds[0:20]:
		print "*"*45
		print "\n"
		print "Predicted: " + topics_to_use[i]
		print classif_data["test_tokens"][count]
		count += 1

		print "\n"

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

	nb.fit(train_set, topic_labels)

	#Now we vectorise each of the test entries and try to label them
	# vect_test_sets = []
	# for i in classif_data["test_tokens"]:
	# 	vect_test_sets.append(vect.transform([i]).toarray())

	# test_set = []
	# for i in vect_test_sets:
	# 	test_set.append(i[0])
	# pp.pprint(test_set[0])
	preds = nb.predict(test_set)
	# pp.pprint(preds)

	# count = 0
	# for i in preds[0:20]:
	# 	print "*"*45
	# 	print "\n"
	# 	print "Predicted: " + topics_to_use[i]
	# 	print classif_data["test_tokens"][count]
	# 	count += 1

	# 	print "\n"
	for i in preds[0:20]:
		print "*"*45
		print "\n"
		print "Predicted: " + topics_to_use[i]
		#print articles[i]["test_tokens"][count]

		print "\n"

def build_run_DecTree(classif_data, vect_data):
	dectree = tree.DecisionTreeClassifier()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	dectree.fit(vect_data["train_vect"], classif_data["topics"])

	#Now we vectorise each of the test entries and try to label them
	vect_test_sets = []
	for i in classif_data["test_tokens"]:
		vect_test_sets.append(vect.transform([i]).toarray())

	test_set = []
	for i in vect_test_sets:
		test_set.append(i[0])
	pp.pprint(test_set[0])
	preds = dectree.predict(test_set)
	pp.pprint(preds)

	count = 0
	for i in preds[0:20]:
		print "*"*45
		print "\n"
		print "Predicted: " + topics_to_use[i]
		print classif_data["test_tokens"][count]
		count += 1

		print "\n"

def build_run_RF(classif_data, vect_data):
	rf = RandomForestClassifier()

	#First, we need to arrange our data 
	#This involves grabbing the train data and labels
	vect = vect_data["vectorizer"]

	rf.fit(vect_data["train_vect"], classif_data["topics"])

	#Now we vectorise each of the test entries and try to label them
	vect_test_sets = []
	for i in classif_data["test_tokens"]:
		vect_test_sets.append(vect.transform([i]).toarray())

	test_set = []
	for i in vect_test_sets:
		test_set.append(i[0])
	pp.pprint(test_set[0])
	preds = rf.predict(test_set)
	pp.pprint(preds)

	count = 0
	for i in preds[0:20]:
		print "*"*45
		print "\n"
		print "Predicted: " + topics_to_use[i]
		print classif_data["test_tokens"][count]
		count += 1

		print "\n"
