from .BaseStep import BaseStep
from ..data.Posts import Posts

import tomotopy as tp
import pandas as pd
import json
import csv

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import warnings
warnings.filterwarnings("ignore")

class PostProcessing(BaseStep):
    
    def __init__(self):
        super().__init__('Post-processing')

        self.__modelFile = 'results/model.bin'
        self.__model = None

        self.__experimentsFile = 'results/experiments.csv'
        self.__experiments = None

        self.__posts = Posts(preProcessed=True, memory=False, splitted=True)

        self.__topicsFile = 'results/topics.csv'
        self.__topicsFields = ['topic', 'label', 'words']

        self.__generalPopularityFile = 'results/general-popularity.csv'
        self.__userPopularityFile = 'results/user-popularity.csv'
        self.__popularityFields = ['user', 'topic', 'year', 'month', 'absolutePopularity', 'relativePopularity']
    
    def __createCSV(self, csvName, fields):
        with open(csvName, 'w', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fields)
            writer.writeheader()
    
    def __appendToCSV(self, csvName, data):
        with open(csvName, 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=data.keys())
            writer.writerow(data)

    def __findLabels(self):
        print('  Extracting topics')

        # Create CSV
        self.__createCSV(self.__topicsFile, self.__topicsFields)

        for topic in range(self.__model.k):
            self.__appendToCSV(
                self.__topicsFile,
                {
                    'topic': topic,
                    'label': 'unknown',
                    'words': ', '.join([ t[0] for t in self.__model.get_topic_words(topic) ]),
                }
            )
        
    
    def __createCoherenceChart(self):
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

        ax.set_title('Best experiment: iterations={} topics={} coherence={:.4f}'.format(Y_max, X_max, Z_max), pad=25)
        ax.set_xlabel('Topics')
        ax.set_ylabel('Iterations')
        ax.set_zlabel('Coherence')

        fig.colorbar(surface, shrink=0.5, aspect=5)
        plt.savefig('results/Coherence-Chart.png')
        plt.clf()
    
    def __createPerplexityChart(self):
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

        ax.text(X_max, Y_max, Z_max, 'V', fontsize=8)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        fig.colorbar(surface, shrink=0.5, aspect=5)

        ax.set_title('Best experiment: iterations={} topics={} perplexity={:.0f}'.format(X_max, Y_max, Z_max), pad=20)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Topics')
        ax.set_zlabel('Perplexity')

        plt.savefig('results/Perplexity-Chart.png')
        plt.clf()
    
    def __normalizeTopics(self, topics):
        normalizer = 1 / float( sum([ weight for _, weight in topics ]) )
        return [ (topic, weight*normalizer) for topic, weight in topics ]
    
    def __getTopics(self, topicDistribution, threshold=0.1):
        # Map topics
        topics = list(zip(range(len(topicDistribution)), topicDistribution))
        topics.sort(key=lambda value: value[1], reverse=True)

        # Check if all topics are below the threshold
        if topics[0][1] < threshold:
            self.__countEmpty += 1
            return self.__normalizeTopics([ (topic, weight) for topic, weight in topics if weight == topics[0][1] ])

        # Remove topics below threshold and normalize
        return self.__normalizeTopics([ (topic, weight) for topic, weight in topics if weight < threshold ])
        
    def __computeUserPopularity(self):
        print('  Computing user popularity')

        numPosts = len(self.__model.docs)
        userDates = {}
        popularityCalculation = {}
        index = -1

        for post in self.__posts:
            # Stop when reach the last post in the model
            if index+1 == numPosts:
                break
            # Compute the metrics if content is not empty
            elif len(post['content']) > 0:
                index += 1
                print('    Posts covered:', index+1, end='\r')

                # Get data
                content = post['content']
                user = post['user']
                date = post['date'].split('-')
                year, month  = (date[0], date[1])

                # Update user dates
                if not user in userDates.keys():
                    userDates[user] = {(year, month): 1}
                elif not (year, month) in userDates[user].keys():
                    userDates[user][(year, month)] = 1
                else:
                    userDates[user][(year, month)] += 1
                
                # Get post topics
                content = self.__model.docs[index]
                topics = self.__getTopics(content.get_topic_dist())

                # Count and sum weight for each topic
                for topic, weight in topics:
                    # Define redis keys
                    countKey = 'count'+user+str(topic)+year+month
                    weightSumKey = 'weight'+user+str(topic)+year+month

                    # Update post counting for this topic
                    if not countKey in popularityCalculation.keys():
                        popularityCalculation[countKey] = 1
                    else:
                        popularityCalculation[countKey] += 1
                    
                    # Update post weight summation for this topic
                    if not weightSumKey in popularityCalculation.keys():
                        popularityCalculation[weightSumKey] = weight
                    else:
                        popularityCalculation[weightSumKey] += weight
        # Print some results
        print('\n    Number of users:', len(userDates.keys()))
        print('    Number of posts with empty topics:', self.__countEmpty)

        # Finishing relative popularity calculation
        self.__createCSV(self.__userPopularityFile, self.__popularityFields)
        calculatedCount = 0
        for user in userDates.keys():
            for year, month in userDates[user].keys():
                monthCount = userDates[user][(year, month)]
                for topic in range(self.__model.k):
                    # Print elapsed metric
                    calculatedCount += 1
                    print('    Computed metrics:', calculatedCount, end='\r')

                    # Define redis keys
                    countKey = 'count'+user+str(topic)+year+month
                    weightSumKey = 'weight'+user+str(topic)+year+month

                    # Check if keys exist
                    if not countKey in popularityCalculation.keys() and not weightSumKey in popularityCalculation.keys():
                        continue

                    # Get computed values
                    count = popularityCalculation[countKey]
                    weightSum = popularityCalculation[weightSumKey]

                    # Compute populatities
                    absolutePopularity = count
                    relativePopularity = weightSum / monthCount

                    # Insert popularity to database
                    self.__appendToCSV(
                        self.__userPopularityFile,
                        {
                            'user': user,
                            'topic': topic,
                            'year': year,
                            'month': month,
                            'absolutePopularity': absolutePopularity,
                            'relativePopularity': relativePopularity,
                        }
                    )
        print()
    
    def _process(self):
        self.__experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)
        self.__experiment = self.__experiments.iloc[self.__experiments.coherence.idxmax()]
        self.__countEmpty = 0
        
        self.__model = tp.LDAModel.load(self.__modelFile)
        self.__findLabels()

        self.__createCoherenceChart()
        self.__createPerplexityChart()
        
        self.__computeUserPopularity()