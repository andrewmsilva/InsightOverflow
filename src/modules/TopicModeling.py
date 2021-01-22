from modules.Step import Step, time
from modules.Data import PreProcessedContents
from modules.Corpus import Corpus
from modules.TopicModel import TopicModel

import pandas as pd
import tomotopy as tp

class TopicModeling(Step):
    
    def __init__(self):
        super().__init__('Topic modeling')

        self.__corpusFile = 'results/corpus.bin'
        self.__corpus = None

        self.__modelFile = 'results/model.bin'

        self.__experiments = pd.DataFrame(columns=['num_topics', 'iterations', 'perplexity', 'coherence'])
        self.__experimentsFile = 'results/experiments.csv'

    # Experiment methods

    def __formatExecutionTime(self, execution_time):
        step = Step()
        step.setExcecutionTime(execution_time)
        return step.getFormatedExecutionTime()
    
    def __buildCorpus(self):
        start_time = time()

        try:
            self.__corpus = tp.utils.Corpus.load(self.__corpusFile)
        except:
            self.__corpus = tp.utils.Corpus()
            for post in PreProcessedContents(splitted=True):
                self.__corpus.add_doc(post)
            #self.__corpus.save(self.__corpusFile)
        
        execution_time = self.__formatExecutionTime(time()-start_time)
        print('  Corpus built: {}'.format(execution_time))

    def __saveExperiment(self, model, num_topics, iterations, perplexity, coherence):
        # Save model with greatest coherence
        if self.__experiments.empty or self.__experiments.iloc[self.__experiments['coherence'].idxmax()]['coherence'] < coherence:
            model.save(self.__modelFile)
        
        # Save experiment to CSV
        row = {
            'num_topics': num_topics,
            'iterations': iterations,
            'perplexity': perplexity,
            'coherence': coherence
        }

        self.__experiments = self.__experiments.append(row, ignore_index=True)
        self.__experiments.to_csv(self.__experimentsFile)
    
        print('  Experiment done: k={} i={} | p={:.2f} cv={:.2f}'.format(num_topics, iterations, perplexity, coherence))
    
    def __runExperiments(self):
        max_topics = 100
        max_iterations = 500

        # Initializing models
        models = [
            tp.LDAModel(corpus=self.__corpus, k=num_topics, min_df=200, rm_top=20, seed=10)
            for num_topics in range(10, max_topics+1, 10)
        ]

        # Start iteration experiments
        for iterations in range(10, max_iterations+1, 10):
            # Train each model and save experiment
            for model in models:
                model.train(iter=10, workers=50)
                cv = tp.coherence.Coherence(model, coherence='c_v')
                self.__saveExperiment(model, model.k, model.global_step, model.perplexity, cv.get_score())
        
    def _process(self):
        self.__buildCorpus()
        self.__runExperiments()
