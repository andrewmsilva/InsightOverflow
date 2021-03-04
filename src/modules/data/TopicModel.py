from .Corpus import Corpus

from gensim.models import LdaMulticore, CoherenceModel
from gensim.models.nmf import Nmf

class TopicModel(object):
    def __init__(self, corpus=None):
        self.__corpus = None
        self.__modelName = None

        self.__model = None
        self.__modelFile = 'results/model.bin'

        self.__coherenceModel = None

    def setCorpus(self, corpus):
        self.__corpus = corpus
    
    def getCoherence(self):
        return self.__coherenceModel.get_coherence()
    
    def getDocumentTopics(self, document, threshold=None):
        return self.__model.get_document_topics(document, threshold)
    
    def build(self, model_name, num_topics, chunksize, passes, corpus=None):
        self.__modelName = model_name
        # Update corpus if necessary
        if corpus:
            self.__corpus = corpus
        # Build topic model
        if model_name == 'lda':
            self.__buildLDA(num_topics, chunksize, passes)
        elif model_name == 'nmf':
            self.__buildNMF(num_topics, chunksize, passes)
        # Build coherence model
        self.__buildCoherenceModel()
    
    def __buildLDA(self, num_topics, chunksize, passes):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        self.__model = LdaMulticore(
            self.__corpus,
            id2word=self.__corpus.getDictionary(),
            num_topics=num_topics,
            chunksize=chunksize,
            passes=passes,
            eval_every=None,
            workers=7,
            random_state=10
        )
    
    def __buildNMF(self, num_topics, chunksize, passes):
        self.__model = Nmf(
            self.__corpus,
            id2word=self.__corpus.getDictionary(),
            num_topics=num_topics,
            chunksize=chunksize,
            passes=passes,
            eval_every=None,
            random_state=10
        )

    def __buildCoherenceModel(self):
        self.__coherenceModel = CoherenceModel(
            model=self.__model,
            texts=self.__corpus.getTexts(),
            coherence='c_v',
            processes=7
        )
    
    def __printTopics(self):
        print('  Topics')
        for idx, topic in self.__model.print_topics(-1):
            print('    {}: {}'.format(idx, topic))
    
    def save(self):
        self.__model.save(self.__modelFile)
    
    def load(self, model_name):
        self.__modelName = model_name

        if model_name == 'lda':
            self.__model = LdaMulticore.load(self.__modelFile)
        elif model_name == 'nmf':
            self.__model = Nmf.load(self.__modelFile)