import pickle

dogs = {"a":[1,2,3], "b": {"c":1}}


cats = pickle.load(open('token_stash.p', 'rb'))

print len(cats)

print cats[140]["topics"]