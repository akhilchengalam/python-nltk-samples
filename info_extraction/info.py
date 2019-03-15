import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags


f = open('processed-image.txt', 'r')
text = f.read()

# sent = nltk.word_tokenize(text)
# sent = nltk.pos_tag(sent)

# print('\n#################  PART OF SPEECH TAGGING  #################\n\n', sent)

# pattern = 'NP: {<DT>?<JJ>*<NN>}'
# cp = nltk.RegexpParser(pattern)
# cs = cp.parse(sent)
# print('\n\n\n##################  OUTPUT  #####################\n\n')
# print(cs)

#Sentence tokenize the input text and apply part of speech
def ie_preprocess(document):
	sentences = nltk.sent_tokenize(document)
	sentences = [nltk.word_tokenize(sent) for sent in sentences]
	sentences = [nltk.pos_tag(sent) for sent in sentences]
	return sentences

sentences = ie_preprocess(text)
print(sentences)

#Chunking
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = []

for sentence in sentences:
	result.append(cp.parse(sentence))

print('\n\n\n################### CHUNK  #####################\n\n')
print(result)

for rst in result:
	rst.draw()
