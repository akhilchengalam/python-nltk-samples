import spacy
# from spacy import displacy
from collections import Counter
import en_core_web_sm


nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
print('\n\n**************Entities**************\n')
print([(X.text, X.label_) for X in doc.ents])

print('\n\n\n\n***********Tokens**************\n')
print([(X, X.ent_iob_, X.ent_type_) for X in doc])