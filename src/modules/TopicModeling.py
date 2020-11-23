from modules.Step import Step
from modules.StreamData import PreProcessedContents

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel, CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.models.nmf import Nmf

import pandas as pd

class Corpus(object):

    def __init__(self, tfidf, bow, length):
        self.__tfidf = tfidf
        self.__bow = bow
        self.__len = length
        
    def __iter__(self):
        for content in self.__bow():
            yield self.__tfidf[content]

    def __len__(self):
        return self.__len

class TopicModeling(Step):
    
    def __init__(self):
        super().__init__('Topic modeling')

        self.__contents = PreProcessedContents(splitted=True)
        self.__dictionary = None
        self.__tfidf = None
        self.__corpus = None

    # Corpus methods

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
        self.__corpus = Corpus(self.__tfidf, self.__buildBow, len(self.__contents))
    
    def __buildCorpus(self, *args):
        self.__buildDictionary(*args)
        self.__buildTfidf()

    # Topic modeling methods

    def __buildLDA(self, num_topics):
        model = LdaModel(
            self.__corpus,
            id2word=self.__dictionary,
            num_topics=num_topics,
            random_state=10
        )
        return model
    
    def __buildMalletLDA(self, num_topics):
        mallet_path = "modules/mallet-2.0.8/bin/"
        model = LdaMallet(
            mallet_path,
            corpus=self.__corpus,
            id2word=self.__dictionary,
            num_topics=num_topics,
            random_seed=10
        )
        return model
    
    def __buildNMF(self, num_topics):
        model = Nmf(
            self.__corpus,
            id2word=self.__dictionary,
            num_topics=num_topics,
            random_state=10
        )
        return model
    
    def __buildTopicModel(self, model_name, *args):
        model = None
        if model_name == 'lda':
            model = self.__buildLDA(*args)
        elif model_name == 'mallet':
            model = self.__buildMalletLDA(*args)
        elif model_name == 'nmf':
            model = self.__buildNMF(*args)
        return model

    def __computeCoherence(self, model):
        coherence_model = CoherenceModel(model=model, texts=self.__contents, coherence='c_v')
        coherence = coherence_model.get_coherence()
        return coherence
    
    def __printTopics(self, model):
        print('  Topics')
        for idx, topic in model.print_topics(-1):
            print('    {}: {}'.format(idx, topic))
    
    # Main method
    
    def _process(self):
        # Dictionary parameters
        no_below_list = [20]
        no_above_list = [0.4]
        keep_n_list = [4000]
        # Topic model parameters
        num_topics_list = [10, 20, 40, 60, 80]
        model_name_list = ['lda', 'nmf']

        count = 0
        for no_below in no_below_list:
            for no_above in no_above_list:
                for keep_n in keep_n_list:
                    # Build dictionary, bag-of-words, TF-IDF, and corpus
                    self.__buildCorpus(no_below, no_above, keep_n)
                    for num_topics in num_topics_list:
                        for model_name in model_name_list:
                            model = self.__buildTopicModel(model_name, num_topics)
                            coherence = self.__computeCoherence(model)

                            print('Experiment', count, 'presented coherence', coherence)
                            count += 1
