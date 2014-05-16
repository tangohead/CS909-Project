import pickle

f = open("token_stash2.p", "rb")
dataset = pickle.load(f)
f.close()

count = 0
arts = 0
for i in dataset:
	if i["train"] == False:
		arts += 1
		if len(i["topics"]) == 0:
			count += 1

print "Test articles with no topic: " + str(count)
print "Test articles: " + str(arts)