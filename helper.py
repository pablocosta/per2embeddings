import unicodedata
import re
import torch
from torch.autograd import Variable
import time
import math
import string
from unicodedata import normalize
import random
import numpy as np
SOS_token = 0
EOS_token = 1
PAD_token = 2
use_cuda = torch.cuda.is_available()




class AverageMeter(object):
    """
    Computes and stores the average and current value
    Borrowed from ImageNet training in PyTorch project
    https://github.com/pytorch/examples/tree/master/imagenet
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return cleaned


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: 'PAD'}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').        read().strip().split('\n')




    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('\t')] for l in lines]


    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, max_length, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
def get_word_embeddings(embedding_size, pre_trained_embeddings, vocab):
    words = []
    vectors = []
    word2idx = {}

    with open(pre_trained_embeddings, 'r') as f:
        idx = 0
        for l in f:
            line = l.strip().split(" ")
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array([i.replace(",",".") for i in line[1:]]).astype(np.float)
            vectors.append(vect)

    glove = {w: vectors[word2idx[w]] for w in words}
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, embedding_size))
    words_found = 0

    for key, value in vocab.items():
        try:
            weights_matrix[value] = glove[key]
            words_found += 1
        except KeyError:
            weights_matrix[value] = np.random.normal(scale=0.6, size=(embedding_size,))
    return weights_matrix


def variableFromSentence(lang, sentence, max_length):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    indexes.extend([PAD_token] * (max_length - len(indexes)))
    result = torch.LongTensor(indexes)
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPairs(input_lang, output_lang, pairs, max_length):
    res = []
    for pair in pairs:
        input_variable = variableFromSentence(input_lang, pair[0], max_length)
        target_variable = variableFromSentence(output_lang, pair[1], max_length)
        res.append((input_variable, target_variable))
    return res



def tensor2np(tensor):
    return tensor.data.cpu().numpy()