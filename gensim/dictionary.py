import os
import gensim
import numpy as np
from gensim import models
from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open

# Create gensim dictionary form a single tet file
dictionary = corpora.Dictionary(simple_preprocess(line, deacc=True) for line in open('processed-image.txt', encoding='utf-8'))

#Dictionary created from file
print('\t\t\t\t###########DICTIONARY###########\n')
print(dictionary.token2id)

f = open('processed-image.txt', 'r')
my_docs = [f.read()]

# Tokenize the docs
tokenized_list = [simple_preprocess(doc) for doc in my_docs]

# Create the Corpus
mydict = corpora.Dictionary()
mycorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_list]
print('\n\n\n\t\t\t\t###########CORPUS###########\n')
print(mycorpus)

word_counts = [[(dictionary[id], count) for id, count in line] for line in mycorpus]
print('\n\n\n\n\t\t\t\t###########WORD COUNT###########\n')
print(word_counts)

dictionary.save('dictionary.dict')  # save dict to disk
corpora.MmCorpus.serialize('bow_corpus.mm', mycorpus)  # save corpus to disk

# Create the TF-IDF model
tfidf = models.TfidfModel(mycorpus, smartirs='ntc')

# Show the TF-IDF weights
print('\n\n\n\n\t\t\t\t###########TF-IDF WEIGHTS###########\n')
for doc in tfidf[mycorpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])