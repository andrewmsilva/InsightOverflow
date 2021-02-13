from modules.Step import Step
from modules.Data import Users, Dates, PreProcessedContents

import tomotopy as tp
import pandas as pd
import json

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import warnings
warnings.filterwarnings("ignore")

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
    
    def __coherenceChart(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X = self.__experiments['num_topics'].tolist()
        Y = self.__experiments['iterations'].tolist()
        Z = self.__experiments['coherence'].tolist()

        Z_max = max(Z)
        index = Z.index(Z_max)
        X_max = X[index]
        Y_max = Y[index]

        surface = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

        ax.text(X_max, Y_max, Z_max, 'V', fontsize=12)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.set_title('Best experiment: iterations={} topics={} coherence={:.4f}'.format(Y_max, X_max, Z_max), pad=20)
        ax.set_xlabel('Topics')
        ax.set_ylabel('Iterations')
        ax.set_zlabel('Coherence')

        fig.colorbar(surface, shrink=0.5, aspect=5)
        plt.savefig('results/Coherence-Chart.png')
        plt.clf()
    
    def __perplexityChart(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X = self.__experiments['iterations'].tolist()
        Y = self.__experiments['num_topics'].tolist()
        Z = self.__experiments['perplexity'].tolist()

        Z_max = min(Z)
        index = Z.index(Z_max)
        X_max = X[index]
        Y_max = Y[index]

        surface = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

        ax.text(X_max, Y_max, Z_max, 'V', fontsize=12)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        fig.colorbar(surface, shrink=0.5, aspect=5)

        ax.set_title('Best experiment: iterations={} topics={} perplexity={:.0f}'.format(X_max, Y_max, Z_max), pad=20)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Topics')
        ax.set_zlabel('Perplexity')

        plt.savefig('results/Perplexity-Chart.png')
        plt.clf()

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
    
    def __getTopics(self, topic_distribution, threshold=0.1):
        topics = list(zip(range(len(topic_distribution)), topic_distribution))
        topics.sort(key=lambda value: value[1])

        for i in range(len(topics)-1, -1, -1):
            topic, weight = topics[i]
            if weight < threshold:
                del topics[i]
            normalizer = 1 / float( sum([ weight for _, weight in topics ]) )
            topics = [ (topic, weight*normalizer) for topic, weight in topics ]
        
        return topics

    def __computeMetrics(self):
        self.__initMetrics()
        # Compute measures
        num_posts = len(self.__model.docs)
        data = zip(self.__posts, self.__users)
        for (post, user) in data:
            if self.__generalPopularity['count'] == num_posts:
                break
            elif len(post) > 0:
                # Counting posts for general popularity
                self.__generalPopularity['count'] += 1
                print('  Posts covered:', self.__generalPopularity['count'], end='\r')
                # Counting posts for user popularity
                users = [ user_['user'] for user_ in self.__userPopularity ]
                user_i = users.index(user) if user in users else None
                if not user_i:
                    user_i = len(self.__userPopularity)
                    self.__userPopularity.append(self.__topicMetrics)
                    self.__userPopularity[user_i]['user'] = user
                self.__userPopularity[user_i]['count'] += 1
                # Getting post topics
                post = self.__model.docs[self.__generalPopularity['count']-1]
                topics = self.__getTopics(post.get_topic_dist())
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
        self.__experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)
        self.__experiment = self.__experiments.iloc[self.__experiments.coherence.idxmax()]
        
        self.__model = tp.LDAModel.load(self.__modelFile)
        self.__printTopics()

        self.__coherenceChart()
        self.__perplexityChart()
        
        self.__computeMetrics()
        self.__saveMetrics()
        self.__printResults()