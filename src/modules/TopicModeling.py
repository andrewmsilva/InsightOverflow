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

        self.__modelFile = 'results/model.bin'
        self.__experimentsFile = 'results/experiments.csv'
        self.__experiments = None

    # Experiment methods

    def __formatExecutionTime(self, execution_time):
        step = Step()
        step.setExcecutionTime(execution_time)
        return step.getFormatedExecutionTime()
    
    def __addCorpus(self, model):
        for post in PreProcessedContents(splitted=True):
            if len(post) > 0:
                model.add_doc(post)
    
    def __trainModels(self):
        max_topics = 100
        max_iterations = 1000

        # Start experiments
        for iterations in range(10, max_iterations+1, 10):
            for num_topics in range(10, max_topics+1, 10):
                # Create model and add corpus
                model = tp.LDAModel(k=num_topics, min_df=200, rm_top=20, seed=10)
                self.__addCorpus(model)
                # Train model
                model.train(iter=iterations, workers=40)
                # Compute c_v coherence
                cv = tp.coherence.Coherence(model, coherence='c_v')
                # Save experiment
                self.__saveExperiment(model, cv.get_score())
        
    def __saveExperiment(self, model, coherence):
        # Save model with greatest coherence
        if self.__experiments.empty or self.__experiments.iloc[self.__experiments['coherence'].idxmax()]['coherence'] < coherence:
            model.save(self.__modelFile, full=False)
        
        # Save experiment to CSV
        row = {
            'iterations': model.global_step,
            'num_topics': model.k,
            'perplexity': model.perplexity,
            'coherence': coherence
        }

        self.__experiments = self.__experiments.append(row, ignore_index=True)
        self.__experiments.to_csv(self.__experimentsFile)
    
        print('  Experiment done: i={} k={} | p={:.2f} cv={:.2f}'.format(row['iterations'], row['num_topics'], row['perplexity'], row['coherence']))
    
    def _process(self):
        self.__experiments = pd.DataFrame(columns=['iterations', 'num_topics', 'perplexity', 'coherence'])
        self.__trainModels()
