from .BaseStep import BaseStep, time
from ..data.Posts import Posts

import pandas as pd
import tomotopy as tp
from multiprocessing import Process
import gc

def split(string):
    return string.split()

class TopicModeling(BaseStep):
    
    def __init__(self):
        super().__init__('Topic modeling')

        self.__posts = Posts(preProcessed=True, memory=False, maxLen=10000)
        self.__posts.contents.itemProcessing = split

        self.__modelFile = 'results/model.bin'
        self.__experimentsFile = 'results/experiments.csv'

    # Experiment methods

    def __formatExecutionTime(self, execution_time):
        step = BaseStep()
        step.setExcecutionTime(execution_time)
        return step.getFormatedExecutionTime()

    def __addCorpus(self, model):
        for content in self.__posts.contents:
            if len(content) > 0:
                model.add_doc(content)
    
    def __trainModel(self, num_topics):
        # Load experiments
        experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)
        # Running if this is a new experiment
        if not (experiments['num_topics'] == num_topics).any():
            # Create model and add corpus
            model = tp.LDAModel(k=num_topics, min_df=200, rm_top=20, seed=10)
            self.__addCorpus(model)
            # Iterate
            for iterations in range(1, 21):
                # Train model
                model.train(iter=1, workers=0)
                # Compute c_v coherence
                cv = tp.coherence.Coherence(model, coherence='c_v')
                coherence = cv.get_score()       
                # Save model if it has the greatest coherence
                if experiments.empty or experiments.iloc[experiments['coherence'].idxmax()]['coherence'] < coherence:
                    model.save(self.__modelFile)
                # Save experiment
                row = [model.k, model.global_step, model.perplexity, coherence]
                experiments = experiments.append(dict(zip(experiments.columns, row)), ignore_index=True)
                experiments.to_csv(self.__experimentsFile)
                # Print result
                print('  Experiment done: k={} i={} p={:.2f} cv={:.2f}'.format(row[0], row[1], row[2], row[3]))  
            # Clear memory
            del model, cv, experiments
            gc.collect()
        return 0  

    def _process(self):
        # Create experiments csv
        try:
            experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)
        except:
            experiments = pd.DataFrame(columns=['num_topics', 'iterations', 'perplexity', 'coherence'])
            experiments.to_csv(self.__experimentsFile)
        
        # Run experiments
        max_topics = 100
        for num_topics in range(10, max_topics+1, 10):
            p = Process(target=self.__trainModel, args=[num_topics])
            p.start()
            p.join()
            p.terminate()

            # Clear memory
            del p
            gc.collect()
        
        # Print best experiment
        experiments = pd.read_csv(self.__experimentsFile, index_col=0, header=0)
        best = experiments.iloc[experiments['coherence'].idxmax()]
        print('  Best experiment: i={} k={} p={:.2f} cv={:.2f}'.format(best['num_topics'], best['iterations'], best['perplexity'], best['coherence']))