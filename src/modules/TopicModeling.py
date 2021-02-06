from modules.Step import Step, time
from modules.Data import PreProcessedContents
from modules.Corpus import Corpus
from modules.TopicModel import TopicModel

import pandas as pd
import tomotopy as tp
from multiprocessing import Process

class TopicModeling(Step):
    
    def __init__(self):
        super().__init__('Topic modeling')

        self.__modelFile = 'results/model.bin'
        self.__experimentsFile = 'results/experiments.csv'

    # Experiment methods

    def __formatExecutionTime(self, execution_time):
        step = Step()
        step.setExcecutionTime(execution_time)
        return step.getFormatedExecutionTime()

    def __addCorpus(self, model):
        for post in PreProcessedContents(splitted=True):
            if len(post) > 0:
                model.add_doc(post)
    
    def __trainModel(self, iterations, num_topics):
        start_time = time()

        # Create model and add corpus
        model = tp.LDAModel(k=num_topics, min_df=200, rm_top=20, seed=10)
        self.__addCorpus(model)
        # Train model
        model.train(iter=iterations, workers=40)
        # Compute c_v coherence
        cv = tp.coherence.Coherence(model, coherence='c_v')
        # Save experiment
        self.__saveExperiment(model, cv.get_score(), start_time)
        
    def __saveExperiment(self, model, coherence, start_time):
        experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)

        # Initialize data
        execution_time = self.__formatExecutionTime(time()-start_time)
        row = [model.global_step, model.k, execution_time, model.perplexity, coherence]

        # Save model with greatest coherence
        if experiments.empty or experiments.iloc[experiments['coherence'].idxmax()]['coherence'] < coherence:
            model.save(self.__modelFile, full=False)
        
        # Save experiments
        experiments = experiments.append(dict(zip(experiments.columns, row)), ignore_index=True)
        experiments.to_csv(self.__experimentsFile)
    
        print('  Experiment done ({:0.2f} GB): i={} k={} t={} p={:.2f} cv={:.2f}'.format(self._getMemoryUsage()/1024**3, row[0], row[1],row[2], row[3], row[4]))      

    def _process(self):
        experiments = pd.DataFrame(columns=['iterations', 'num_topics', 'execution_time', 'perplexity', 'coherence'])
        experiments.to_csv(self.__experimentsFile)

        max_iterations = 1000
        max_topics = 100
        for iterations in range(10, max_iterations+1, 10):
            for num_topics in range(10, max_topics+1, 10):
                p = Process(target=self.__trainModel, args=(iterations, num_topics))
                p.start()
                p.join()
