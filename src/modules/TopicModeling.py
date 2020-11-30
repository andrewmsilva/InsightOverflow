from modules.Step import Step
from modules.Corpus import Corpus
from modules.TopicModel import TopicModel

import pandas as pd

class TopicModeling(Step):
    
    def __init__(self):
        super().__init__('Topic modeling')

        self.__corpus = Corpus()
        self.__model = TopicModel()

        self.__experiments = pd.DataFrame(columns=['no_below', 'no_above', 'keep_n', 'num_topics', 'model_name', 'coherence'])
        self.__experimentsFile = 'results/experiments.csv'

    # Experiment methods

    def __saveExperiment(self, no_below, no_above, keep_n, num_topics, model_name, coherence):
        # Save model with greatest coherence
        if self.__experiments.empty or self.__experiments.iloc[self.__experiments['coherence'].idxmax()]['coherence'] < coherence:
            self.__corpus.save()
            self.__model.save()
        # Save experiment to CSV
        row = {
            'no_below': no_below,
            'no_above': no_above,
            'keep_n': keep_n,
            'num_topics': num_topics,
            'model_name': model_name,
            'coherence': coherence
        }
        self.__experiments = self.__experiments.append(row, ignore_index=True)
        self.__experiments.to_csv(self.__experimentsFile)
    
    def _process(self):
        # Dictionary parameters
        no_below_list = [200]
        no_above_list = [0.2, 0.4]
        keep_n_list = [1000, 2000, 4000]
        # Topic model parameters
        num_topics_list = [10, 20, 40, 60, 80]
        model_name_list = ['lda', 'nmf']
        # Starting experiments
        count = 0
        for no_below in no_below_list:
            for no_above in no_above_list:
                for keep_n in keep_n_list:
                    self.__corpus.build(no_below, no_above, keep_n)
                    for num_topics in num_topics_list:
                        for model_name in model_name_list:
                            # Build topic model and compute coherence
                            self.__model.build(model_name, num_topics, self.__corpus)
                            coherence = self.__model.getCoherence()
                            # Save experiment to file
                            self.__saveExperiment(no_below, no_above, keep_n, num_topics, model_name, coherence)
                            # Print some results
                            print('Experiment', count, 'presented coherence', coherence)
                            count += 1
