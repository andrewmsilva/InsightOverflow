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
        self.__labeledTopicsFile = 'results/labeled-topics.csv'
        self.__topicsFields = ['topic', 'label', 'words']
        self.__labels = None

        self.__generalPopularityFile = 'results/general-popularity.csv'
        self.__generalPopularityFields = ['topic', 'semester', 'popularity']
        self.__generalDriftFile = 'results/general-drift.csv'
        self.__generalDriftFields = ['topic', 'mean', 'variance', 'drift']
        self.__generalTrendsFile = 'results/general-trends.csv'
        self.__generalTrendsFields = ['topic', 'popularity']

        self.__userPopularityFile = 'results/user-popularity.csv'
        self.__userPopularityFields = ['user'] + self.__generalPopularityFields
        self.__userDriftFile = 'results/user-drift.csv'
        self.__userDriftFields = ['user'] + self.__generalDriftFields
        self.__userTrendsFile = 'results/user-trends.csv'
        self.__userTrendsFields = ['user'] + self.__generalTrendsFields
    
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
            df = pd.read_csv(self.__labeledTopicsFile, header=0)
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

        fig.suptitle('Best experiment: iterations={} topics={} coherence={:.4f}'.format(Y_max, X_max, Z_max))
        fig.tight_layout()

        ax.set_xlabel('Number of topics')
        ax.set_ylabel('Iterations')
        ax.set_zlabel('Coherence')

        plt.savefig('results/Coherence-Chart.png', dpi=600)
        plt.clf()
    
    def __normalizeTopics(self, topics):
        if len(topics) == 0:
            return []
        
        normalizer = 1 / float( sum([ weight for _, weight in topics ]) )
        return [ (topic, weight*normalizer) for topic, weight in topics ]
    
    def __getTopics(self, topicDistribution, threshold=0.1):
        # Map topics
        topics = list(zip(range(len(topicDistribution)), topicDistribution))
        topics.sort(key=lambda value: value[1], reverse=True)

        # Remove topics below threshold and normalize
        return self.__normalizeTopics([ (topic, weight) for topic, weight in topics if weight >= threshold ])

    def __initCalculator(self):
        return {'count': 0, 'weightSum': [0]*self.__model.k}
    
    def __saveDrift(self, semesterCount, popularities, csvName, user=None):
        for topic in popularities.keys():
            lengthDifference = semesterCount - len(popularities[topic])

            if lengthDifference > 0:
                popularities[topic] = [0]*lengthDifference + popularities[topic]

            mean = statistics.mean(popularities[topic])
            variance = statistics.pvariance(popularities[topic], mu=mean)
            drift = variance**0.5

            if not user:
                self.__appendToCSV(
                    csvName,
                    {
                        'topic': topic,
                        'mean': mean,
                        'variance': variance,
                        'drift': drift,
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
                    'drift': drift,
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
                # Get topics
                index += 1
                content = self.__model.docs[index]
                topics = self.__getTopics(content.get_topic_dist())

                # Check if post has topics
                if len(topics) == 0:
                    self.__countEmpty += 1
                    continue

                # Get data
                user = post['user']
                year, month, day = post['date'].split('-')
                semester = f'{year}.{1 if int(month) < 7 else 2}'

                # Adjust dict of users and semesters
                if not user in calculation.keys():
                    calculation[user] = {semester: self.__initCalculator()}
                elif not semester in calculation[user].keys():
                    calculation[user][semester] = self.__initCalculator()
                
                # Increment posts counter
                calculation[user][semester]['count'] += 1

                # Sum weight for each topic
                for topic, weight in topics:
                    calculation[user][semester]['weightSum'][topic] += weight
        
        # Print some results
        print('    Number of users:', len(calculation.keys()))
        print('    Posts with empty topics:', self.__countEmpty)

        # Initialize CSVs
        self.__createCSV(self.__userPopularityFile, self.__userPopularityFields)
        self.__createCSV(self.__userDriftFile, self.__userDriftFields)
        self.__createCSV(self.__userTrendsFile, self.__userTrendsFields)

        # Finish relative popularity calculation
        computedCount = 0
        for user in calculation.keys():
            popularities = {}
            trendPopularityCalculation = self.__initCalculator()
            for semester in calculation[user].keys():
                trendPopularityCalculation['count'] += calculation[user][semester]['count']
                for topic in range(self.__model.k):
                    # Check if metric must be computed
                    if calculation[user][semester]['weightSum'][topic] == 0:
                        if topic in popularities.keys():
                            popularities[topic].append(0)
                        continue
                    
                    # Compute populatities
                    trendPopularityCalculation['weightSum'][topic] += calculation[user][semester]['weightSum'][topic]
                    popularity = calculation[user][semester]['weightSum'][topic] / calculation[user][semester]['count']
                    computedCount += 1

                    # Append relative popularity to compute variance
                    if topic not in popularities.keys():
                        popularities[topic] = []
                    popularities[topic].append(popularity)

                    # Insert popularity to csv
                    self.__appendToCSV(
                        self.__userPopularityFile,
                        {
                            'user': user,
                            'topic': topic,
                            'semester': semester,
                            'popularity': popularity,
                        }
                    )
            
            # Insert trend popularity to csv
            trendPopularities = [
                trendPopularityCalculation['weightSum'][topic] / trendPopularityCalculation['count'] 
                for topic in range(self.__model.k)
            ]
            for topic in range(self.__model.k):
                if trendPopularityCalculation['weightSum'][topic] > 0:
                    computedCount += 1
                    popularity = trendPopularityCalculation['weightSum'][topic] / trendPopularityCalculation['count']
                    self.__appendToCSV(
                        self.__userTrendsFile,
                        {
                            'user': user,
                            'topic': topic,
                            'popularity': popularity,
                        }
                    )
            
            # Compute drift
            popularities = { topic: popularities[topic] for topic in sorted(popularities.keys()) }
            self.__saveDrift(len(calculation[user].keys()), popularities, self.__userDriftFile, user)
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
                # Get topics
                index += 1
                content = self.__model.docs[index]
                topics = self.__getTopics(content.get_topic_dist())

                # Check if post has topics
                if len(topics) == 0:
                    self.__countEmpty += 1
                    continue

                # Get semester
                year, month, day = post['date'].split('-')
                semester = f'{year}.{1 if int(month) < 7 else 2}'

                # Adjust dict of semesters
                if not semester in calculation.keys():
                    calculation[semester] = self.__initCalculator()
                
                # Increment posts counter
                calculation[semester]['count'] += 1

                # Sum weight for each topic
                for topic, weight in topics:
                    calculation[semester]['weightSum'][topic] += weight
        
        # Print some results
        print('    Posts with empty topics:', self.__countEmpty)

        # Initialize CSVs
        self.__createCSV(self.__generalPopularityFile, self.__generalPopularityFields)
        self.__createCSV(self.__generalDriftFile, self.__generalDriftFields)
        self.__createCSV(self.__generalTrendsFile, self.__generalTrendsFields)

        # Finish relative popularity calculation
        popularities = {}
        trendPopularityCalculation = self.__initCalculator()
        computedCount = 0
        for semester in calculation.keys():
            trendPopularityCalculation['count'] += calculation[semester]['count']
            for topic in range(self.__model.k):
                # Check if metric must be computed
                if calculation[semester]['weightSum'][topic] == 0:
                    if topic in popularities.keys():
                        popularities[topic].append(0)
                    continue
                
                # Compute populatities
                trendPopularityCalculation['weightSum'][topic] += calculation[semester]['weightSum'][topic]
                popularity = calculation[semester]['weightSum'][topic] / calculation[semester]['count']
                computedCount += 1

                # Append relative popularity to compute variance
                if topic not in popularities.keys():
                    popularities[topic] = []
                popularities[topic].append(popularity)

                # Insert popularity to csv
                self.__appendToCSV(
                    self.__generalPopularityFile,
                    {
                        'topic': topic,
                        'semester': semester,
                        'popularity': popularity,
                    }
                )
        
        # Insert trend popularity to csv
        trendPopularities = [
            trendPopularityCalculation['weightSum'][topic] / trendPopularityCalculation['count'] 
            for topic in range(self.__model.k)
        ]
        for topic in range(self.__model.k):
            if trendPopularityCalculation['weightSum'][topic] > 0:
                computedCount += 1
                popularity = trendPopularityCalculation['weightSum'][topic] / trendPopularityCalculation['count']
                self.__appendToCSV(
                    self.__generalTrendsFile,
                    {
                        'topic': topic,
                        'popularity': popularity,
                    }
                )

        # Compute drift
        popularities = { topic: popularities[topic] for topic in sorted(popularities.keys()) }
        self.__saveDrift(len(calculation.keys()), popularities, self.__generalDriftFile)
        computedCount += 4
        
        print('    Computed metrics:', computedCount)
    
    def __saveChart(self, yLabel, xTicks, path, legends=True):       
        if isinstance(yLabel, str): plt.ylabel(yLabel)
        if legends: plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize='small',  borderaxespad=0, labelspacing=0.8)
        if isinstance(xTicks, Iterable): plt.xticks(xTicks, rotation=45, ha='left')
        plt.tight_layout()
        plt.savefig(path, dpi=600)
        plt.clf()
    
    def __createUserCharts(self):
        print('  Creating user popularity charts')
        
        originalPopularityDf = pd.read_csv(self.__userPopularityFile, header=0)
        originaldriftDf = pd.read_csv(self.__userDriftFile, header=0)
        originalTrendsDf = pd.read_csv(self.__userTrendsFile, header=0)
        users = originaldriftDf.user.unique()

        count = 0
        random.seed(0)
        random.shuffle(users)
        for user in users:
            # Stop 10 users was analyzed
            if count == 10:
                break

            # Skip user if his contribution ir lower than one year
            popularityDf = originalPopularityDf.loc[originalPopularityDf.user == user]
            if len(popularityDf.semester.unique()) < 12:
                continue
            
            count += 1
            driftDf = originaldriftDf.loc[originaldriftDf.user == user]
            trendsDf = originalTrendsDf.loc[originalTrendsDf.user == user]
            semesters = popularityDf.semester.unique()

            popularitiesByMonth = []
            drifts = []
            trendPopularities = []
            topics = []
            for topic in range(int(self.__experiment.num_topics)):
                popularities = []
                for i in range(len(semesters)):
                    semester = semesters[i]
                    popularitiesRows = popularityDf.loc[(popularityDf.semester == semester) & (popularityDf.topic == topic)]
                    
                    if len(popularitiesRows) == 0:
                        popularities.append(0)
                    else:
                        popularities.append(popularitiesRows.iloc[-1].popularity)
                    
                if (any([ value for value in popularities if value != 0 ])):
                    popularitiesByMonth.append(popularities)
                    topics.append(topic)
                
                if topic in topics:
                    driftRows = driftDf.loc[driftDf.topic == topic]
                    drifts.append(driftRows.iloc[-1].drift)

                    trendsRows = trendsDf.loc[trendsDf.topic == topic]
                    trendPopularities.append(trendsRows.iloc[-1].popularity)
            
            # Create palette
            palette = sns.color_palette('muted', 10) + sns.color_palette('colorblind', 10) + sns.color_palette('dark', 10)

            # Load topic labels if possible
            labels = [ self.__labels[i] for i in topics ] if isinstance(self.__labels, list) else topics
            enum = [x for x, _ in enumerate(labels)]

            # Create popularity stacked chart
            plt.rcParams['xtick.labelbottom'] = False
            plt.rcParams['xtick.labeltop'] = True

            plt.figure(figsize=(8,5))
            plt.stackplot(semesters, popularitiesByMonth, labels=labels, colors=palette)
            plt.margins(0,0)
            self.__saveChart('Topic Popularity', semesters, f'results/User-{user}-Popularity-Stacked-Chart.png')

            # Create loayalty bar chart
            plt.rcParams['xtick.labelbottom'] = True
            plt.rcParams['xtick.labeltop'] = False

            plt.figure(figsize=(8,5))
            plt.bar(enum, drifts, color=palette)
            plt.xticks(enum, labels, rotation=45, ha='right')
            plt.margins(0,0)
            self.__saveChart('Popularity drift', None, f'results/User-{user}-Drift-Bar-Chart.png', False)

            # Create trend popularity bar chart
            plt.figure(figsize=(8,5))
            plt.bar(enum, trendPopularities, color=palette)
            plt.xticks(enum, labels, rotation=45, ha='right')
            plt.margins(0,0)
            self.__saveChart('Trend popularity', None, f'results/User-{user}-Trends-Bar-Chart.png', False)

    def __createGeneralCharts(self):
        print('  Creating general popularity charts')

        popularityDf = pd.read_csv(self.__generalPopularityFile, header=0)
        driftDf = pd.read_csv(self.__generalDriftFile, header=0)
        trendsDf = pd.read_csv(self.__generalTrendsFile, header=0)
        semesters = popularityDf.semester.unique()

        popularitiesByMonth = []
        drifts = []
        trendPopularities = []
        topics = []
        for topic in range(int(self.__experiment.num_topics)):
            popularities = []
            for i in range(len(semesters)):
                popularityRows = popularityDf.loc[(popularityDf.semester == semesters[i]) & (popularityDf.topic == topic)]
                if len(popularityRows) == 0:
                    popularities.append(0)
                else:
                    popularities.append(popularityRows.iloc[-1].popularity)
                
            if (any([ value for value in popularities if value != 0 ])):
                popularitiesByMonth.append(popularities)
                topics.append(topic)
            
            if topic in topics:
                driftRows = driftDf.loc[driftDf.topic == topic]
                drifts.append(driftRows.iloc[-1].drift)

                trendsRows = trendsDf.loc[trendsDf.topic == topic]
                trendPopularities.append(trendsRows.iloc[-1].popularity)
        
        # Create palette
        palette = sns.color_palette('muted', 10) + sns.color_palette('colorblind', 10) + sns.color_palette('dark', 10)

        # Load topic labels if possible
        labels = [ self.__labels[i] for i in topics ] if isinstance(self.__labels, list) else topics
        enum = [x for x, _ in enumerate(labels)]

        # Create popularity stacked chart
        plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.labeltop'] = True
        
        plt.figure(figsize=(8,5))
        plt.stackplot(semesters, popularitiesByMonth, labels=labels, colors=palette)
        plt.margins(0,0)
        self.__saveChart('Topic Popularity', semesters, 'results/General-Popularity-Stacked-Chart.png')

        # Create drift bar chart
        plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.labeltop'] = False

        plt.figure(figsize=(8,5))
        plt.bar(enum, drifts, color=palette)
        plt.xticks(enum, labels, rotation=45, ha='right')
        plt.margins(0,0)
        self.__saveChart('Popularity drift', None, 'results/General-Drift-Bar-Chart.png', False)

        # Create trend popularity bar chart
        plt.figure(figsize=(8,5))
        plt.bar(enum, trendPopularities, color=palette)
        plt.xticks(enum, labels, rotation=45, ha='right')
        plt.margins(0,0)
        self.__saveChart('Trend popularity', None, 'results/General-Trends-Bar-Chart.png', False)

        # Create trend lines
        for i in range(len(popularitiesByMonth)):
            topic = topics[i]
            popularities = popularitiesByMonth[i]

            plt.figure(figsize=(12, 2))
            plt.plot(popularities, color=palette[i], linewidth=12)
            plt.axis('off')
            plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
            plt.savefig(f'results/Topic-{topic}-Evolution-Line-Chart.png', dpi=50)
            plt.clf()
    
    def _process(self):
        self.__experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)
        self.__experiments = self.__experiments.astype({'num_topics': 'int32', 'iterations': 'int32', 'perplexity': 'float32', 'coherence': 'float32'})
        
        self.__experiment = self.__experiments.iloc[self.__experiments.coherence.idxmax()]
        self.__countEmpty = 0
        
        # self.__model = tp.LDAModel.load(self.__modelFile)
        # self.__extractTopics()
        self.__loadLabeledTopics()

        self.__createCoherenceChart()

        # self.__computeGeneralPopularity()
        self.__createGeneralCharts()
        
        # self.__computeUserPopularity()
        self.__createUserCharts()
