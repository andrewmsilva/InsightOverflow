from modules.Step import Step, time
from modules.Data import PreProcessedContents
from modules.Corpus import Corpus
from modules.TopicModel import TopicModel

import pandas as pd
import tomotopy as tp
from os import remove

class TopicModeling(Step):
    
    def __init__(self):
        super().__init__('Topic modeling')

        self.__corpus = None

        self.__modelFile = 'results/model.bin'
        self.__model = None

        self.__experimentsFile = 'results/experiments.csv'
        self.__experiments = None

    # Experiment methods

    def __formatExecutionTime(self, execution_time):
        step = Step()
        step.setExcecutionTime(execution_time)
        return step.getFormatedExecutionTime()
    
    def __buildCorpus(self):
        start_time = time()
        self.__corpus = tp.utils.Corpus()
        for post in PreProcessedContents(splitted=True):
            self.__corpus.add_doc(post)
        execution_time = self.__formatExecutionTime(time()-start_time)
        print('  Corpus built: {}'.format(execution_time))
    
    def __trainModels(self):
        max_topics = 100
        max_iterations = 1000

        # Start experiments
        for iterations in range(10, max_iterations+1, 10):
            for num_topics in range(10, max_topics+1, 10):
                # Train and load model
                self.__model = tp.LDAModel(corpus=self.__corpus, k=num_topics, min_df=200, rm_top=20, seed=10)
                self.__model.train(iter=iterations, workers=50)
                # Compute c_v coherence
                cv = tp.coherence.Coherence(self.__model, coherence='c_v')
                # Save experiment
                self.__saveExperiment(cv.get_score())
        
    def __saveExperiment(self, coherence):
        # Save model with greatest coherence
        if self.__experiments.empty or self.__experiments.iloc[self.__experiments['coherence'].idxmax()]['coherence'] < coherence:
            self.__model.save(self.__modelFile, full=False)
        
        # Save experiment to CSV
        row = {
            'iterations': self.__model.global_step,
            'num_topics': self.__model.k,
            'perplexity': self.__model.perplexity,
            'coherence': coherence
        }

        self.__experiments = self.__experiments.append(row, ignore_index=True)
        self.__experiments.to_csv(self.__experimentsFile)
    
        print('  Experiment done: i={} k={} | p={:.2f} cv={:.2f}'.format(row['iterations'], row['num_topics'], row['perplexity'], row['coherence']))
    
    def _process(self):
        self.__experiments = pd.DataFrame(columns=['iterations', 'num_topics', 'perplexity', 'coherence'])
        self.__buildCorpus()
        self.__trainModels()
