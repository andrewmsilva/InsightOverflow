from modules.Step import Step
from modules.Data import Users, Dates, PreProcessedContents

import tomotopy as tp
import pandas as pd
import json

class PostProcessing(Step):
    
    def __init__(self):
        super().__init__('Post-processing')

        self.__modelFile = 'results/model.bin'
        self.__model = None

        self.__experimentsFile = 'results/experiments.csv'
        self.__experiments = None

        self.__users = Users()
        self.__dates = Dates()
        self.__posts = PreProcessedContents(splitted=True)

        self.__generalPopularityFile = 'results/general-popularity.json'
        self.__generalSemmianualPopularityFile = 'results/general-semmianual-popularity.json'
        self.__userPopularityFile = 'results/user-popularity.json'
        self.__userSemmianualPopularityFile = 'results/user-semmianual-popularity.json'

    def __printTopics(self):
        for topic in range(self.__model.k):
            print('  Topic {}: {}'.format(topic, ', '.join([ t[0] for t in self.__model.get_topic_words(topic) ])))
    
    @property
    def __topicMetrics(self):
        return {
            'count': 0,
            'absolute': [0]*self.__experiment.num_topics,
            'relative': [0]*self.__experiment.num_topics
        }
    
    def __initMetrics(self):
        self.__generalPopularity = self.__topicMetrics
        self.__generalSemmianualPopularity = []
        self.__userPopularity = []
        self.__userSemmianualPopularity = []

    def __computeMetrics(self):
        self.__initMetrics()
        # Compute measures
        for (post, user) in zip(self.__posts, self.__users):
            # Counting posts for general popularity
            self.__generalPopularity['count'] += 1
            #print('  Posts covered:', self.__generalPopularity['count'], end='\r')
            # Counting posts for user popularity
            users = [ user_['user'] for user_ in self.__userPopularity ]
            user_i = users.index(user) if user in users else None
            if not user_i:
                user_i = len(self.__userPopularity)
                self.__userPopularity.append(self.__topicMetrics)
                self.__userPopularity[user_i]['user'] = user
            self.__userPopularity[user_i]['count'] += 1
            # Getting post topics
            post = self.__model.make_doc(post)
            print(self.__model.docs[0].get_topic_dist())
            topics, ll = self.__model.infer([post], workers=4)
            break
            for topic, weight in topics:
                # Computing values for general popularity
                self.__generalPopularity['absolute'][topic] += 1
                self.__generalPopularity['relative'][topic] += weight
                # Computing values for user popularity
                self.__userPopularity[user_i]['absolute'][topic] += 1
                self.__userPopularity[user_i]['relative'][topic] += weight
        # Finishing relative popularity calculation
        for topic in range(self.__experiment.num_topics):
            self.__generalPopularity['relative'][topic] /= self.__generalPopularity['count']
            for user_i in range(len(self.__userPopularity)):
                self.__userPopularity[user_i]['relative'][topic] /= self.__userPopularity[user_i]['count']
    
    def __saveMetrics(self):
        with open(self.__generalPopularityFile, 'w') as f:
            f.write(json.dumps(self.__generalPopularity))
        with open(self.__generalSemmianualPopularityFile, 'w') as f:
            f.write(json.dumps(self.__generalSemmianualPopularity))
        with open(self.__userPopularityFile, 'w') as f:
            f.write(json.dumps(self.__userPopularity))
        with open(self.__userSemmianualPopularityFile, 'w') as f:
            f.write(json.dumps(self.__userSemmianualPopularity))
    
    def __printResults(self):
        print('  Number of posts: {}'.format(self.__generalPopularity['count']))
        print('  Number of users: {}'.format(len(self.__userPopularity)))
    
    def _process(self):
        self.__experiments = pd.read_csv(self.__experimentsFile)
        self.__experiment = self.__experiments.iloc[self.__experiments.coherence.idxmax()]
        
        self.__model = tp.LDAModel.load(self.__modelFile)
        self.__printTopics()
        
        self.__computeMetrics()
        self.__saveMetrics()
        self.__printResults()