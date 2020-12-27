from modules.Step import Step, time
from modules.Corpus import Corpus
from modules.TopicModel import TopicModel

import pandas as pd

class TopicModeling(Step):
    
    def __init__(self):
        super().__init__('Topic modeling')

        self.__corpus = None
        self.__model = TopicModel()

        self.__experiments = pd.DataFrame(columns=['model_name', 'num_topics', 'chunksize', 'passes', 'execution_time', 'coherence'])
        self.__experimentsFile = 'results/experiments.csv'

    # Experiment methods

    def __formatExecutionTime(self, execution_time):
        step = Step()
        step.setExcecutionTime(execution_time)
        return step.getFormatedExecutionTime()
    
    def __buildCorpus(self):
        start_time = time()
        self.__corpus = Corpus()
        execution_time = self.__formatExecutionTime(time()-start_time)
        print('  Corpus built: {}'.format(execution_time))

    def __saveExperiment(self, model_name, num_topics, chunksize, passes, execution_time, coherence):
        # Save model with greatest coherence
        if self.__experiments.empty or self.__experiments.iloc[self.__experiments['coherence'].idxmax()]['coherence'] < coherence:
            self.__model.save()        
        # Save experiment to CSV
        execution_time = self.__formatExecutionTime(execution_time)
        row = {
            'model_name': model_name,
            'num_topics': num_topics,
            'chunksize': chunksize,
            'passes': passes,
            'execution_time': execution_time,
            'coherence': coherence
        }
        self.__experiments = self.__experiments.append(row, ignore_index=True)
        self.__experiments.to_csv(self.__experimentsFile)
    
        print('  Experiment done: {}, {}, {}, {} | {}, {:.4f}'.format(model_name, num_topics, chunksize, passes, execution_time, coherence))
    
    def __runExperiments(self):
        passes_list = [10, 100, 200]
        chunksize_list = [5000, 50000, 500000]
        num_topics_list = [20, 40, 60, 80]
        model_name_list = ['lda', 'nmf']
        # Starting experiments
        for passes in passes_list:
            for chunksize in chunksize_list:
                for num_topics in num_topics_list:
                    for model_name in model_name_list:
                        start_time = time()
                        # Build topic model and compute coherence
                        self.__model.build(model_name, num_topics, chunksize, passes, self.__corpus)
                        coherence = self.__model.getCoherence()
                        # Save experiment to file
                        self.__saveExperiment(model_name, num_topics, chunksize, passes, time()-start_time, coherence)

    def _process(self):
        self.__buildCorpus()
        self.__runExperiments()
