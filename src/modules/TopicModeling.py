from modules.Step import Step
from modules.StreamData import PreProcessedContents

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

class Corpus(object):

    def __init__(self, tfidf):
        self.__tfidf = tfidf

    def __iter__(self):
        for i in range(self.__len):
            yield self.__tfidf[bow_corpus[i]]

    def __len__(self):
        return self.__len

class TopicModeling(Step):
    
    def __init__(self):
        super().__init__('Topic modeling')

        self.__contents = PreProcessedContents(splitted=True)
        self.__dictionary = None
        self.__lda = None
        self.__tfidf = None
        self.__corpus = None
        self.__experiments = []