from .BaseStep import BaseStep
from ..data.Posts import Posts

import tomotopy as tp
import pandas as pd
import numpy as np
import json
import csv
import random
import statistics
from collections import Iterable 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import seaborn as sns

sns.set_style('whitegrid')

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
        self.__labaledTopicsFile = 'results/labeled-topics.csv'
        self.__topicsFields = ['topic', 'label', 'words']
        self.__labels = None

        self.__generalPopularityFile = 'results/general-popularity.csv'
        self.__generalPopularityFields = ['topic', 'date', 'popularity']
        self.__generalLoyaltyFile = 'results/general-loyalty.csv'
        self.__generalLoyaltyFields = ['topic', 'mean', 'variance', 'standardDeviation', 'trend']

        self.__userPopularityFile = 'results/user-popularity.csv'
        self.__userPopularityFields = ['user'] + self.__generalPopularityFields
        self.__userLoyaltyFile = 'results/user-loyalty.csv'
        self.__userLoyaltyFields = ['user'] + self.__generalLoyaltyFields
    
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
    
    def __loadLabeledTopics(self):
        try:
            df = pd.read_csv(self.__labaledTopicsFile, header=0)
            self.__labels = df.label.tolist()
        except:
            pass
    
    def __createCoherenceChart(self):
        print('  Creating coherence chart')

        months = self.__experiments['num_topics'].tolist()
        popularities = self.__experiments['iterations'].tolist()
        Z = self.__experiments['coherence'].tolist()

        Z_max = max(Z)
        index = Z.index(Z_max)
        X_max = months[index]
        Y_max = popularities[index]

        fig = plt.figure(figsize=(8,5))
        ax = fig.gca(projection='3d')
        surface = ax.plot_trisurf(months, popularities, Z, cmap=cm.coolwarm, linewidth=0)

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

        months = self.__experiments['iterations'].tolist()
        popularities = self.__experiments['num_topics'].tolist()
        Z = self.__experiments['perplexity'].tolist()

        Z_max = min(Z)
        index = Z.index(Z_max)
        X_max = months[index]
        Y_max = popularities[index]

        fig = plt.figure(figsize=(8,5))
        ax = fig.gca(projection='3d')
        surface = ax.plot_trisurf(months, popularities, Z, cmap=cm.coolwarm, linewidth=0)

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
        return {'count': 0, 'weightSum': [0]*self.__model.k}
    
    def __saveLoyalty(self, datesCount, popularities, csvName, user=None):
        for topic in popularities.keys():
            lengthDifference = datesCount - len(popularities[topic])

            if lengthDifference > 0:
                popularities[topic] = [0]*lengthDifference + popularities[topic]

            mean = statistics.mean(popularities[topic])
            variance = statistics.pvariance(popularities[topic], mu=mean)
            standardDeviation = variance**0.5

            if not user:
                self.__appendToCSV(
                    csvName,
                    {
                        'topic': topic,
                        'mean': mean,
                        'variance': variance,
                        'standardDeviation': standardDeviation,
                    }
                )
            else:
                self.__appendToCSV(
                csvName,
                {
                    'user': user,
                    'topic': topic,
                    'mean': mean,
                    'variance': variance,
                    'standardDeviation': standardDeviation,
                }
            )
    
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
                index += 1
                content = self.__model.docs[index]
                topics = self.__getTopics(content.get_topic_dist())

                # Sum weight for each topic
                for topic, weight in topics:
                    calculation[user][date]['weightSum'][topic] += weight
        
        # Print some results
        print('    Number of users:', len(calculation.keys()))
        print('    Posts with empty topics:', self.__countEmpty)

        # Initialize CSVs
        self.__createCSV(self.__userPopularityFile, self.__userPopularityFields)
        self.__createCSV(self.__userLoyaltyFile, self.__userLoyaltyFields)

        # Finish relative popularity calculation
        computedCount = 0
        for user in calculation.keys():
            popularities = {}
            for date in calculation[user].keys():
                for topic in range(self.__model.k):
                    # Check if metric must be computed
                    if calculation[user][date]['weightSum'][topic] == 0:
                        if topic in popularities.keys():
                            popularities[topic].append(0)
                        continue
                    
                    # Compute populatities
                    popularity = calculation[user][date]['weightSum'][topic] / calculation[user][date]['count']
                    computedCount += 1

                    # Append relative popularity to compute variance
                    if topic not in popularities.keys():
                        popularities[topic] = []
                    popularities[topic].append(popularity)

                    # Insert popularity to database]
                    self.__appendToCSV(
                        self.__userPopularityFile,
                        {
                            'user': user,
                            'topic': topic,
                            'date': date,
                            'popularity': popularity,
                        }
                    )
            popularities = { topic: popularities[topic] for topic in sorted(popularities.keys()) }
            self.__saveLoyalty(len(calculation[user].keys()), popularities, self.__userLoyaltyFile, user)
            computedCount += 4
        
        print('    Computed metrics:', computedCount)
    
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
                # Get data
                content = post['content']
                date = post['date'][:7]

                # Adjust dict of dates
                if not date in calculation.keys():
                    calculation[date] = self.__initCalculator()
                
                # Increment posts counter
                calculation[date]['count'] += 1
                
                # Get post topics
                index += 1
                content = self.__model.docs[index]
                topics = self.__getTopics(content.get_topic_dist())

                # Sum weight for each topic
                for topic, weight in topics:
                    calculation[date]['weightSum'][topic] += weight
        
        # Print some results
        print('    Posts with empty topics:', self.__countEmpty)

        # Initialize CSVs
        self.__createCSV(self.__generalPopularityFile, self.__generalPopularityFields)
        self.__createCSV(self.__generalLoyaltyFile, self.__generalLoyaltyFields)

        # Finish relative popularity calculation
        popularities = {}
        computedCount = 0
        for date in calculation.keys():
            for topic in range(self.__model.k):
                # Check if metric must be computed
                if calculation[date]['weightSum'][topic] == 0:
                    if topic in popularities.keys():
                        popularities[topic].append(0)
                    continue
                
                # Compute populatities
                popularity = calculation[date]['weightSum'][topic] / calculation[date]['count']
                computedCount += 1

                # Append relative popularity to compute variance
                if topic not in popularities.keys():
                    popularities[topic] = []
                popularities[topic].append(popularity)

                # Insert popularity to database
                self.__appendToCSV(
                    self.__generalPopularityFile,
                    {
                        'topic': topic,
                        'date': date,
                        'popularity': popularity,
                    }
                )
        popularities = { topic: popularities[topic] for topic in sorted(popularities.keys()) }
        self.__saveLoyalty(len(calculation.keys()), popularities, self.__generalLoyaltyFile)
        computedCount += 4
        
        print('    Computed metrics:', computedCount)
    
    def __getXTicks(self, months):
        if len(months) <= 20:
            return months
        
        xticks = []
        count = 0
        for date in months:
            if count == 6:
                   count = 0
            count += 1
            if count == 1:
                xticks.append(date)
            else:
                xticks.append('')
        return xticks
    
    def __saveChart(self, yLabel, xTicks, path, legends=True):       
        plt.ylabel(yLabel)
        if legends: plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize='small',  borderaxespad=0, labelspacing=0.8)
        if isinstance(xTicks, Iterable): plt.xticks(xTicks, rotation=45, ha='left', position=(1,1))
        plt.tight_layout()
        plt.savefig(path, dpi=600)
        plt.clf()
    
    def __getUsersWithAtLeastOneYear(self, df):
        candidates = []
        users = df.user.unique()
        for user in users:
            userDf = df.loc[df.user == user]
            monthsCount = len(userDf.date.unique())
            if monthsCount >= 12:
                candidates.append(user)
        
        return candidates
    
    def __createUserCharts(self):
        print('  Creating user popularity charts')
        
        originalPopularityDf = pd.read_csv(self.__userPopularityFile, header=0)
        originalLoyaltyDf = pd.read_csv(self.__userLoyaltyFile, header=0)

        users = self.__getUsersWithAtLeastOneYear(originalPopularityDf)
        print(f'    Users with at least one year of contribution: {len(users)}')

        random.seed(10)
        for user in random.sample(users, 5):
            popularityDf = originalPopularityDf.loc[originalPopularityDf.user == user]
            loyaltyDf = originalLoyaltyDf.loc[originalLoyaltyDf.user == user]
            months = popularityDf.date.unique()

            popularitiesByMonth = []
            standardDeviations = []
            topics = []
            for topic in range(int(self.__experiment.num_topics)):
                popularities = []
                for i in range(len(months)):
                    date = months[i]
                    popularitiesRows = popularityDf.loc[(popularityDf.date == date) & (popularityDf.topic == topic)]
                    
                    if len(popularitiesRows) == 0:
                        popularities.append(0)
                    else:
                        popularities.append(popularitiesRows.iloc[-1].popularity)
                    
                if (any([ value for value in popularities if value != 0 ])):
                    popularitiesByMonth.append(popularities)
                    topics.append(topic)
                
                if topic in topics:
                    loyaltyRows = loyaltyDf.loc[loyaltyDf.topic == topic]
                    standardDeviations.append(loyaltyRows.iloc[-1].standardDeviation)
            
            # Create palette
            palette = sns.color_palette('muted', len(topics))

            # Load topic labels if possible
            labels = [ self.__labels[i] for i in topics ] if isinstance(self.__labels, list) else topics

            # Create Stacked chart
            plt.rcParams['xtick.labelbottom'] = False
            plt.rcParams['xtick.labeltop'] = True

            plt.figure(figsize=(8,5))
            plt.stackplot(months, popularitiesByMonth, labels=labels, colors=palette)
            plt.margins(0,0)
            self.__saveChart('Topic Popularity', self.__getXTicks(months), f'results/User-{user}-Popularity-Stacked-Chart.png')

            # Create bar chart
            plt.rcParams['xtick.labelbottom'] = True
            plt.rcParams['xtick.labeltop'] = False

            plt.figure(figsize=(8,5))
            enum = [x for x, _ in enumerate(labels)]
            plt.bar(enum, standardDeviations, color=palette)
            plt.xticks(enum, labels, rotation=45, ha='right')
            plt.margins(0,0)
            self.__saveChart('Standard deviation', None, f'results/User-{user}-Loyalty-Bar-Chart.png', False)

    def __createGeneralCharts(self):
        print('  Creating general popularity charts')

        popularityDf = pd.read_csv(self.__generalPopularityFile, header=0)
        loyaltyDf = pd.read_csv(self.__generalLoyaltyFile, header=0)
        months = popularityDf.date.unique()

        popularitiesByMonth = []
        standardDeviations = []
        topics = []
        for topic in range(int(self.__experiment.num_topics)):
            popularities = []
            for i in range(len(months)):
                popularityRows = popularityDf.loc[(popularityDf.date == months[i]) & (popularityDf.topic == topic)]
                if len(popularityRows) == 0:
                    popularities.append(0)
                else:
                    popularities.append(popularityRows.iloc[-1].popularity)
                
            if (any([ value for value in popularities if value != 0 ])):
                popularitiesByMonth.append(popularities)
                topics.append(topic)
            
            if topic in topics:
                loyaltyRows = loyaltyDf.loc[loyaltyDf.topic == topic]
                standardDeviations.append(loyaltyRows.iloc[-1].standardDeviation)
        
        # Create palette
        palette = sns.color_palette('muted', len(topics))

        # Load topic labels if possible
        labels = [ self.__labels[i] for i in topics ] if isinstance(self.__labels, list) else topics

        # Create stacked chart
        plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.labeltop'] = True
        
        plt.figure(figsize=(8,5))
        plt.stackplot(months, popularitiesByMonth, labels=labels, colors=palette)
        plt.margins(0,0)
        self.__saveChart('Topic Popularity', self.__getXTicks(months), 'results/General-Popularity-Stacked-Chart.png')

        # Create bar chart
        plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.labeltop'] = False

        plt.figure(figsize=(8,5))
        enum = [x for x, _ in enumerate(labels)]
        plt.bar(enum, standardDeviations, color=palette)
        plt.xticks(enum, labels, rotation=45, ha='right')
        plt.margins(0,0)
        self.__saveChart('Standard deviation', None, 'results/General-Loyalty-Bar-Chart.png', False)
    
    def _process(self):
        self.__experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)
        self.__experiments = self.__experiments.astype({'num_topics': 'int32', 'iterations': 'int32', 'perplexity': 'float32', 'coherence': 'float32'})
        
        self.__experiment = self.__experiments.iloc[self.__experiments.coherence.idxmax()]
        self.__countEmpty = 0
        
        # self.__model = tp.LDAModel.load(self.__modelFile)
        # self.__extractTopics()
        self.__loadLabeledTopics()

        self.__createCoherenceChart()
        self.__createPerplexityChart()

        # self.__computeGeneralPopularity()
        self.__createGeneralCharts()
        
        # self.__computeUserPopularity()
        self.__createUserCharts()
