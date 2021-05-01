from .BaseStep import BaseStep
from ..data.Posts import Posts

import tomotopy as tp
import pandas as pd
import json
import csv
import random

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns

sns.set_style('whitegrid')
palette = plt.get_cmap('Set1')

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
        self.__generalPopularityFields = ['topic', 'date', 'absolutePopularity', 'relativePopularity']
        self.__generalPopularityDf = None

        self.__userPopularityFile = 'results/user-popularity.csv'
        self.__userPopularityFields = ['user', 'topic', 'date', 'absolutePopularity', 'relativePopularity']
        self.__userPopularityDf = None
    
    def __createCSV(self, csvName, fields):
        with open(csvName, 'w', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fields)
            writer.writeheader()
    
    def __appendToCSV(self, csvName, data):
        with open(csvName, 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=data.keys())
            writer.writerow(data)

    def __extractTopics(self):
        print('  Extracting topics')

        # Create CSV
        self.__createCSV(self.__topicsFile, self.__topicsFields)

        for topic in range(self.__model.k):
            self.__appendToCSV(
                self.__topicsFile,
                {
                    'topic': topic,
                    'label': 'unknown',
                    'words': ' '.join([ t[0] for t in self.__model.get_topic_words(topic) ]),
                }
            )
    
    def __createCoherenceChart(self):
        print('  Creating coherence chart')

        X = self.__experiments['num_topics'].tolist()
        Y = self.__experiments['iterations'].tolist()
        Z = self.__experiments['coherence'].tolist()

        Z_max = max(Z)
        index = Z.index(Z_max)
        X_max = X[index]
        Y_max = Y[index]

        fig = plt.figure(figsize=(8,5))
        ax = fig.gca(projection='3d')
        surface = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        fig.colorbar(surface, shrink=0.5, aspect=5)

        fig.suptitle('Coherence by iterations and number of topics\nBest experiment: iterations={} topics={} coherence={:.4f}'.format(Y_max, X_max, Z_max))
        fig.tight_layout()

        ax.set_xlabel('Number of topics')
        ax.set_ylabel('Iterations')
        ax.set_zlabel('Coherence')

        plt.savefig('results/Coherence-Chart.png', dpi=300)
        plt.clf()
    
    def __createPerplexityChart(self):
        print('  Creating perplexity chart')

        X = self.__experiments['iterations'].tolist()
        Y = self.__experiments['num_topics'].tolist()
        Z = self.__experiments['perplexity'].tolist()

        Z_max = min(Z)
        index = Z.index(Z_max)
        X_max = X[index]
        Y_max = Y[index]

        fig = plt.figure(figsize=(8,5))
        ax = fig.gca(projection='3d')
        surface = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        fig.colorbar(surface, shrink=0.5, aspect=5)

        fig.suptitle('Perplexity by iterations and number of topics\nBest experiment: iterations={} topics={} perplexity={:.0f}'.format(X_max, Y_max, Z_max))
        fig.tight_layout()

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Number of topics')
        ax.set_zlabel('Perplexity')

        plt.savefig('results/Perplexity-Chart.png', dpi=300)
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
        return self.__normalizeTopics([ (topic, weight) for topic, weight in topics if weight >= threshold ])

    def __initCalculator(self):
        return {'count': 0, 'topicCount': [0]*self.__model.k, 'topicWeightSum': [0]*self.__model.k}
    
    def __computeUserPopularity(self):
        print('  Computing user popularity')

        self.__countEmpty = 0
        numPosts = len(self.__model.docs)
        calculation = {}
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
                date = post['date'][:7]

                # Adjust dict of users and dates
                if not user in calculation.keys():
                    calculation[user] = {date: self.__initCalculator()}
                elif not date in calculation[user].keys():
                    calculation[user][date] = self.__initCalculator()
                
                # Increment posts counter
                calculation[user][date]['count'] += 1
                
                # Get post topics
                content = self.__model.docs[index]
                topics = self.__getTopics(content.get_topic_dist())

                # Count and sum weight for each topic
                for topic, weight in topics:
                    # Increment post counting for this topic
                    calculation[user][date]['topicCount'][topic] += 1
                    
                    # Update post weight summation for this topic
                    calculation[user][date]['topicWeightSum'][topic] += weight
        
        # Print some results
        print('\n    Number of users:', len(calculation.keys()))
        print('    Number of posts with empty topics:', self.__countEmpty)

        # Finishing relative popularity calculation
        self.__createCSV(self.__userPopularityFile, self.__userPopularityFields)
        calculatedCount = 0
        for user in calculation.keys():
            for date in calculation[user].keys():
                for topic in range(self.__model.k):
                    # Print elapsed metric
                    calculatedCount += 1
                    print('    Computed metrics:', calculatedCount, end='\r')

                    # Check if keys exist
                    if calculation[user][date]['topicCount'][topic] == 0:
                        continue

                    # Compute populatities
                    absolutePopularity = calculation[user][date]['topicCount'][topic]
                    relativePopularity = calculation[user][date]['topicWeightSum'][topic] / calculation[user][date]['count']

                    # Insert popularity to database
                    self.__appendToCSV(
                        self.__userPopularityFile,
                        {
                            'user': user,
                            'topic': topic,
                            'date': date,
                            'absolutePopularity': absolutePopularity,
                            'relativePopularity': relativePopularity,
                        }
                    )
        print()
    
    def __computeGeneralPopularity(self):
        print('  Computing general popularity')

        self.__countEmpty = 0
        numPosts = len(self.__model.docs)
        calculation = {}
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
                date = post['date'][:7]

                # Adjust dict of dates
                if not date in calculation.keys():
                    calculation[date] = self.__initCalculator()
                
                # Increment posts counter
                calculation[date]['count'] += 1
                
                # Get post topics
                content = self.__model.docs[index]
                topics = self.__getTopics(content.get_topic_dist())

                # Count and sum weight for each topic
                for topic, weight in topics:
                    # Increment post counting for this topic
                    calculation[date]['topicCount'][topic] += 1
                    
                    # Update post weight summation for this topic
                    calculation[date]['topicWeightSum'][topic] += weight
        
        # Print some results
        print('\n    Number of posts with empty topics:', self.__countEmpty)

        # Finishing relative popularity calculation
        self.__createCSV(self.__generalPopularityFile, self.__generalPopularityFields)
        calculatedCount = 0
        for date in calculation.keys():
            for topic in range(self.__model.k):
                # Print elapsed metric
                calculatedCount += 1
                print('    Computed metrics:', calculatedCount, end='\r')

                # Check if keys exist
                if calculation[date]['topicCount'][topic] == 0:
                    continue

                # Compute populatities
                absolutePopularity = calculation[date]['topicCount'][topic]
                relativePopularity = calculation[date]['topicWeightSum'][topic] / calculation[date]['count']

                # Insert popularity to database
                self.__appendToCSV(
                    self.__generalPopularityFile,
                    {
                        'topic': topic,
                        'date': date,
                        'absolutePopularity': absolutePopularity,
                        'relativePopularity': relativePopularity,
                    }
                )
        print()
    
    def __getXTicks(self, X):
        if len(X) <= 20:
            return X
        
        xticks = []
        count = 0
        for date in X:
            if count == 6:
                   count = 0
            count += 1
            if count == 1:
                xticks.append(date)
            else:
                xticks.append('')
        return xticks
    
    def __saveChart(self, X, title, xLabel, yLabel, path):       
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, ncol=2)
        plt.xticks(self.__getXTicks(X), rotation=45)
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.clf()
    
    def __getUsersWithAtLeastOneYear(self, df):
        users = df.user.unique()
        return [ user for user in users if len(list(df.loc[df.user == user].date.unique())) >= 12 ]
    
    def __createUserPopularityCharts(self):
        print('  Creating user popularity charts')
        
        originalDf = pd.read_csv(self.__userPopularityFile, header=0)
        users = self.__getUsersWithAtLeastOneYear(originalDf)
        print(f'    Users with at least one year of contribution: {len(users)}')

        random.seed(10)
        for user in random.sample(users, 3):
            df = originalDf.loc[originalDf.user == user]
            X = df.date.unique()
            plt.figure(figsize=(8,5))

            stackedY = []
            labelY = []
            for topic in range(int(self.__experiment.num_topics)):
                Y = []
                for i in range(len(X)):
                    date = X[i]
                    rows = df.loc[(df.date == date) & (df.topic == topic)]
                    
                    if len(rows) == 0:
                        Y.append(0)
                    else:
                        Y.append(rows.iloc[-1].relativePopularity)
                    
                if (any([ value for value in Y if value != 0 ])):
                    plt.plot(X, Y, marker='', color=palette(topic), linewidth=1, alpha=0.9, label=topic)
                    stackedY.append(Y)
                    labelY.append(topic)
            
            self.__saveChart(X, f'Relative topic popularity by month for user {user}', 'Month', 'Topic Popularity', f'results/User-{user}-Relative-Popularity-Line-Chart.png')

            plt.figure(figsize=(8,5))
            plt.stackplot(X, stackedY, labels=labelY)
            plt.margins(0,0)
            self.__saveChart(X, f'Relative topic popularity by month for user {user}', 'Month', 'Topic Popularity', f'results/User-{user}-Relative-Popularity-Stacked-Chart.png')

            plt.figure(figsize=(8,5))
            stackedY = []
            labelY = []
            for topic in range(int(self.__experiment.num_topics)):
                Y = []
                for i in range(len(X)):
                    date = X[i]
                    rows = df.loc[(df.date == date) & (df.topic == topic)]
                    
                    if len(rows) == 0:
                        Y.append(0)
                    else:
                        Y.append(int(rows.iloc[-1].absolutePopularity))

                if (any([ value for value in Y if value != 0 ])):
                    plt.plot(X, Y, marker='', color=palette(topic), linewidth=1, alpha=0.9, label=topic)
                    stackedY.append(Y)
                    labelY.append(topic)
            
            self.__saveChart(X, f'Absolute topic popularity by month for user {user}', 'Month', 'Number of posts', f'results/User-{user}-Absolute-Popularity-Line-Chart.png')

            plt.figure(figsize=(8,5))
            plt.stackplot(X, stackedY, labels=labelY)
            plt.margins(0,0)
            self.__saveChart(X, f'Absolute topic popularity by month for user {user}', 'Month', 'Number of posts', f'results/User-{user}-Absolute-Popularity-Stacked-Chart.png')

    def __createGeneralPopularityCharts(self):
        print('  Creating general popularity charts')

        df = pd.read_csv(self.__generalPopularityFile, header=0)
        X = df.date.unique()
        plt.figure(figsize=(8,5))

        stackedY = []
        labelY = []
        for topic in range(int(self.__experiment.num_topics)):
            Y = []
            for i in range(len(X)):
                date = X[i]
                rows = df.loc[(df.date == date) & (df.topic == topic)]
                
                if len(rows) == 0:
                    Y.append(0)
                else:
                    Y.append(rows.iloc[-1].relativePopularity)
                
            if (any([ value for value in Y if value != 0 ])):
                plt.plot(X, Y, marker='', color=palette(topic), linewidth=1, alpha=0.9, label=topic)
                stackedY.append(Y)
                labelY.append(topic)

        self.__saveChart(X, 'Relative topic popularity by month in Stack Overflow', 'Month', 'Topic Popularity', 'results/General-Relative-Popularity-Line-Chart.png')

        plt.figure(figsize=(8,5))
        plt.stackplot(X, stackedY, labels=labelY)
        plt.margins(0,0)
        self.__saveChart(X, 'Relative topic popularity by month in Stack Overflow', 'Month', 'Topic Popularity', 'results/General-Relative-Popularity-Stacked-Chart.png')

        plt.figure(figsize=(8,5))
        stackedY = []
        labelY = []
        for topic in range(int(self.__experiment.num_topics)):
            Y = []
            for i in range(len(X)):
                date = X[i]
                rows = df.loc[(df.date == date) & (df.topic == topic)]
                
                if len(rows) == 0:
                    Y.append(0)
                else:
                    Y.append(int(rows.iloc[-1].absolutePopularity))
            
            if (any([ value for value in Y if value != 0 ])):
                plt.plot(X, Y, marker='', color=palette(topic), linewidth=1, alpha=0.9, label=topic)
                stackedY.append(Y)
                labelY.append(topic)
        
        self.__saveChart(X, 'Absolute topic popularity by month in Stack Overflow', 'Month', 'Number of posts', 'results/General-Absolute-Popularity-Line-Chart.png')

        plt.figure(figsize=(8,5))
        plt.stackplot(X, stackedY, labels=labelY)
        plt.margins(0,0)
        self.__saveChart(X, 'Absolute topic popularity by month in Stack Overflow', 'Month', 'Number of posts', 'results/General-Absolute-Popularity-Stacked-Chart.png')
    
    def _process(self):
        self.__experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)
        self.__experiments = self.__experiments.astype({'num_topics': 'int32', 'iterations': 'int32', 'perplexity': 'float32', 'coherence': 'float32'})
        
        self.__experiment = self.__experiments.iloc[self.__experiments.coherence.idxmax()]
        self.__countEmpty = 0
        
        # self.__model = tp.LDAModel.load(self.__modelFile)
        # self.__extractTopics()

        self.__createCoherenceChart()
        self.__createPerplexityChart()

        # self.__computeGeneralPopularity()
        self.__createGeneralPopularityCharts()
        
        # self.__computeUserPopularity()
        self.__createUserPopularityCharts()
