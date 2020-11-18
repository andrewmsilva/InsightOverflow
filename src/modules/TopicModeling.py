from modules.Step import Step
from modules.StreamData import PreProcessedContents

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
from gensim.models.wrappers import LdaMallet

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
    
    def __buildDictionary(self, no_below=None, no_above=None, keep_n=None):
        self.__dictionary = Dictionary(self.__contents)
        if no_below != None and no_above != None and keep_n != None:
            self.__dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

    def __buildBow(self):
        if self.__dictionary != None:
            for content in self.__contents:
                yield self.__dictionary.doc2bow(content)
    
    def __buildTfidf(self):
        self.__tfidf = TfidfModel(self.__buildBow())
        self.__corpus = Corpus(tfidf)

    def __buildLda(self, num_topics):
        model = LdaModel(
            self.__corpus,
            id2word=self.__dictionary,
            num_topics=num_topics,
            random_seed=10
        )
        return model
    
    def __buildMalletLda(self, num_topics):
        mallet_path = "modules/mallet-2.0.8/bin/"
        model = LdaMallet(
            mallet_path,
            corpus=self.__corpus,
            id2word=self.__dictionary,
            num_topics=num_topics,
            random_seed=10
        )
        return model