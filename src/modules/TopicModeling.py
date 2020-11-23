from modules.Step import Step
from modules.StreamData import PreProcessedContents

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel, CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.models.nmf import Nmf

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
        self.__corpus = Corpus(self.__tfidf, self.__buildBow, len(self.__contents))

    def __buildLda(self, num_topics):
        model = LdaModel(
            self.__corpus,
            id2word=self.__dictionary,
            num_topics=num_topics,
            random_state=10
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
            model = self.__buildLda(*args)
        elif model_name == 'mallet':
            model = self.__buildMalletLda(*args)
        elif model_name == 'nmf':
            model = self.__build(*args)
        return model

    def __computeCoherence(self, model):
        coherence_model = CoherenceModel(model=model, texts=self.__contents, coherence='c_v')
        coherence = coherence_model.get_coherence()
        return coherence
    
    def __printTopics(self, model):
        print('  Topics')
        for idx, topic in model.print_topics(-1):
            print('    {}: {}'.format(idx, topic))
    
    def __runExperiment(self, no_below, no_above, keep_n, num_topics, model_name):
        print('Staring experiment:', no_below, no_above, keep_n, num_topics, model_name)

        print('Building dictionary')
        self.__buildDictionary(no_below, no_above, keep_n)

        print('Building TF-IDF')
        self.__buildTfidf()

        print('Building topic model')
        model = self.__buildTopicModel(model_name, num_topics)
        self.__printTopics(model)

        print('Building coherence')
        coherence = self.__computeCoherence(model)

        self.__experiments.append(((no_below, no_above, keep_n, num_topics, model), coherence))
        print(self.__experiments[-1], '\n')
    
    def _process(self):
        # Dictionary parameters
        no_below_list = [20]
        no_above_list = [0.4]
        keep_n_list = [4000]
        num_topics_list = [20]
        model_list = ['lda']
        for no_below in no_below_list:
            for no_above in no_above_list:
                for keep_n in keep_n_list:
                    for num_topics in num_topics_list:
                        for model in model_list:
                            self.__runExperiment(no_below, no_above, keep_n, num_topics, model)