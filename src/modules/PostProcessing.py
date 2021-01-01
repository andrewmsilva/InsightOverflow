from modules.Step import Step
from modules.Data import Users, Dates
from modules.Corpus import Corpus
from modules.TopicModel import TopicModel

import pandas as pd

class PostProcessing(Step):
    
    def __init__(self):
        super().__init__('Post-processing')
        self.__experimentsFile = 'results/experiments.csv'
        self.__users = Users()
        self.__dates = Dates()

        self.__corpus = Corpus()
        self.__model = TopicModel()

        self.__experiments = pd.read_csv(self.__experimentsFile)
        self.__experiment = self.__experiments.iloc[self.__experiments.coherence.idxmax()]
        
        self.__model.load(self.__experiment.model_name)
        self.__model.setCorpus(self.__corpus)

        self.__generalAbsolutePopularity = None
        self.__generalRelativePopularity = None

        self.__generalSemmianualAbsolutePopularity = None
        self.__generalSemmianualRelativePopularity = None

        self.__userAbsolutePopularity = None
        self.__userRelativePopularity = None

        self.__userSemmianualAbsolutePopularity = None
        self.__userSemmianualRelativePopularity = None
    
    def _process(self):
        # Initialize results
        self.__generalAbsolutePopularity = [0]*self.__experiment.num_topics
        self.__generalRelativePopularity = [0]*self.__experiment.num_topics
        # Compute measures
        for document in self.__corpus:
            topics = self.__model.getDocumentTopics(document, 0.1)
            for topic, weight in topics:
                self.__generalAbsolutePopularity[topic] += 1
                self.__generalRelativePopularity[topic] += weight
        for topic in range(self.__experiment.num_topics):
            self.__generalRelativePopularity[topic] /= len(self.__corpus)

        print(self.__generalAbsolutePopularity)
        print(self.__generalRelativePopularity)