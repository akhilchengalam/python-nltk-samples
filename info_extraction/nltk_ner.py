import nltk
import re
import time

exampleArray = ['The incredibly intimidating NLP scares people away who are sissies.']


contentArray =['Mohanlal made his acting debut as a teenager in the Malayalam film Thiranottam in 1978, but the film was delayed in its release for 25 years due to censorship issues.',
               'His screen debut was in the 1980 romance film Manjil Virinja Pookkal, in which he played the villain.',
               'In the following years, he played antagonistic characters in several films and gradually rose to supporting roles.',
               'In 2010, he acted in five films, the first being Janakan, a crime thriller in which he co-starred with Suresh Gopi, written by S. N Swamy.',
               ' In the film, he played Adv. Surya Narayanan who encounters some runaway suspects as they approach him for justice',
               'Mohanlal has been a goodwill ambassador for the government and other nonprofit organisations, mainly for public service ads and humanitarian causes.',
               'In September 2013, the direct-broadcast satellite television provider Tata Sky announced Mohanlal as its brand endorser for its Kerala market'
              ]



##let the fun begin!##
def processLanguage():
    try:
        for item in contentArray:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            print(tagged)

            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw()

            time.sleep(1)

    except Exception as e:
        print(str(e))
        

processLanguage()