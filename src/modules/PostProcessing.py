from modules.Step import Step
from modules.Data import Users, Dates
from modules.Corpus import Corpus
from modules.TopicModel import TopicModel

import pandas as pd
import json

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

        self.__generalPopularityFile = 'results/general-popularity.json'
        self.__generalPopularity = {
            'absolute': None,
            'relative': None,
        }

        self.__generalSemmianualPopularityFile = 'results/general-semmianual-popularity.json'
        self.__generalSemmianualPopularity = {
            'absolute': None,
            'relative': None,
        }

        self.__userPopularityFile = 'results/user-popularity.json'
        self.__userPopularity = {
            'absolute': None,
            'relative': None,
        }

        self.__userSemmianualPopularityFile = 'results/user-semmianual-popularity.json'
        self.__userSemmianualPopularity = {
            'absolute': None,
            'relative': None,
        }

    def __computeMetrics(self):
        # Initialize results
        self.__generalPopularity['absolute'] = [0]*self.__experiment.num_topics
        self.__generalPopularity['relative'] = [0]*self.__experiment.num_topics
        # Compute measures
        for document in self.__corpus:
            topics = self.__model.getDocumentTopics(document, 0.1)
            for topic, weight in topics:
                self.__generalPopularity['absolute'][topic] += 1
                self.__generalPopularity['relative'][topic] += weight
        for topic in range(self.__experiment.num_topics):
            self.__generalPopularity['relative'][topic] /= len(self.__corpus)
    
    def __saveMetrics(self):
        with open(self.__generalPopularityFile, 'w') as f:
            f.write(json.dumps(self.__generalSemmianualPopularity))
        with open(self.__generalSemmianualPopularityFile, 'w') as f:
            f.write(json.dumps(self.__generalSemmianualPopularity))
        with open(self.__userPopularityFile, 'w') as f:
            f.write(json.dumps(self.__userPopularity))
        with open(self.__userSemmianualPopularityFile, 'w') as f:
            f.write(json.dumps(self.__userSemmianualPopularity))
    
    def _process(self):
        self.__computeMetrics()
        self.__saveMetrics()