import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint


ex ="Mohanlal made his acting debut as a teenager in the Malayalam film Thiranottam in 1978, but the film was delayed in its release for 25 years due to censorship issues.'His screen debut was in the 1980 romance film Manjil Virinja Pookkal, in which he played the villain.'In the following years, he played antagonistic characters in several films and gradually rose to supporting roles.'In 2010, he acted in five films, the first being Janakan, a crime thriller in which he co-starred with Suresh Gopi, written by S. N Swamy.' In the film, he played Adv. Surya Narayanan who encounters some runaway suspects as they approach him for justice'Mohanlal has been a goodwill ambassador for the government and other nonprofit organisations, mainly for public service ads and humanitarian causes.'In September 2013, the direct-broadcast satellite television provider Tata Sky announced Mohanlal as its brand endorser for its Kerala market"
              

def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(ex)
print('*****Tokenization and Part of Speech tagging*****\n')
print(sent)

#Noun phrase chunking to identify named entities using a regular expression 
#consisting of rules that indicate how sentences should be chunked.
pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print('\n\n*************Noun Phrase Chunking**************')
print(cs)

tagging = tree2conlltags(cs)
pprint(tagging)

#Recognize named entities
ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
print('\n\n\n***************Named Entities******************\n')
print(ne_tree)