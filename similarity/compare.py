import gensim
from gensim.matutils import softcossim 
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess


f = open('insurance_vocabulary.txt', 'r')
doc1 = f.read()

f = open('processed-image.txt', 'r')
doc2 = f.read()
documents = [doc1, doc2]

fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

# Prepare a dictionary and a corpus.
dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

# Prepare the similarity matrix
similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

sent_1 = dictionary.doc2bow(simple_preprocess(doc1))
sent_2 = dictionary.doc2bow(simple_preprocess(doc2))

sentences = [sent_1, sent_2]

print(softcossim(sent_1, sent_2, similarity_matrix))