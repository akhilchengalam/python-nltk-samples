import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm


nlp = en_core_web_sm.load()
# import pdb
# pdb.set_trace()
f = open('processed-image.txt', 'r')
text = f.read().replace('\n\n', '').replace('\n', ',')
article = nlp(text)

labels = [x.label_ for x in article.ents]
print('\n\n*************Label Counts**************\n')
print(Counter(labels))

sentences = [x for x in article.sents]

print(dict([(str(x), x.label_) for x in article.ents]))
