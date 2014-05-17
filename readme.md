A range of packages are needed to run this system. It requires the latest version of:
+ Natural Language Toolkit (NLTK) (including the NLTK corpus)
+ GenSim
+ Numpy
+ Scikit-Learn
+ Beautiful Soup 4 
+ Scipy

To run the system, use 
```
    python load_data.py A B C D
```
Parameters
==========
A is the _mode_:
+ 1 - Bag of words (unigram) feature testing
+ 2 - Bigram feature testing
+ 3 - Trigram feature testing
+ 4 - Bag of words (unigram) topic model feature testing
+ 5 - Bigram topic model feature testing
+ 6 - Trigram topic model feature testing
+ 7 - Bag of words (unigram) clustering
+ 8 - Bag of words (unigram) test classification

Note that option 7 runs and evaluates all three clustering algorithms

B is the _classification mode_, indicating the type of classifier to use for that feature mode, and only applies to options 1-6:
+ 1 - Naive Bayes classifier
+ 2 - Decision Tree classifier
+ 3 - Random Forests classifier

Each is run with via K-fold cross-validation, where k=10

C indicates whether to limit _data loading_:
+ 1 - Limit data loading
+ 0 - Do not limit data loading

This is used in conjunction with D, which is the _number_ of Reuters sub-files to load, from 1-21. If 0 is passed as C, this parameter should take the value 0.

Notes
========
+ Options C and D are largely for debugging purposes.
+ The system is likely to take a long time to complete its task, especially if it is run without loading limits. Expect multiple hours to run the ngram classifiers. Topics models are typically much faster.
+ wrapper.py provides a wrapper to run modes 1 to 6 with all three classifiers. This takes a **long** time to run!
+ Output during runtime is inserted into output.txt in the _scripts_ directory.
