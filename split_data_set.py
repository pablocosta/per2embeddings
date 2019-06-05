
# -*- coding: utf-8 -*-
import os
import re
import string
from unicodedata import normalize
from sklearn.model_selection import train_test_split

# clean a list of lines
def clean_pairs(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			line = [w.replace('txt', "") for w in line]
			# remove tokens with numbers in them
			#line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return cleaned

corpus_name = "b5"
corpus = os.path.join("./data", corpus_name)

def printLines(aqr, n=10):

	with open(aqr) as datafile:

		for i, line in enumerate(datafile):
			line = line.strip()
			if i == n:
				break
			print(line)


printLines(os.path.join(corpus, "corpus.txt"))





def loadLines(fileName):
    lines = []
    with open(fileName, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip().split("\t"))
    lines = clean_pairs(lines)

    return lines

# Define path to new file
datafile = os.path.join(corpus, "corpus.txt")

delimiter = '\t'




# Load lines and process conversations
print("\nProcessing corpus...")
pairs = loadLines(os.path.join(datafile))


pairs_train, pairs_val = train_test_split(pairs, test_size=0.2, random_state=42)

train_caption = open("./train_caption.pt", "w+")
train_description = open("./train_description.pt", "w+")
train_personality = open("./train_personality.pt", "w+")
for pairs in pairs_train:

	train_caption.write("".join(pairs[-1]) +"\n")
	train_description.write("".join(pairs[-2]) +"\n")
	train_personality.write("".join(pairs[1]) +"\n")

train_caption.close()
train_description.close()


train_caption = open("./val_caption.pt", "w+")
train_description = open("./val_description.pt", "w+")
train_personality = open("./val_personality.pt", "w+")
for pairs in pairs_val:
	train_caption.write("".join(pairs[-1]) + "\n")
	train_description.write("".join(pairs[-2]) + "\n")
	train_personality.write("".join(pairs[1]) + "\n")

train_description.close()
train_caption.close()



