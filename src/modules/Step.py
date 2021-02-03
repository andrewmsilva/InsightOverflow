from time import time
import os, psutil

class Step(object):

    def __init__(self, name=None):
        self.__name = 'Unamed step'
        if type(name) == str:
            self.__name = name
        
        self.__executionTime = None
        self.__process = psutil.Process(os.getpid())
    
    def getName(self):
        return self.__name
    
    def setExcecutionTime(self, execution_time):
        self.__executionTime = execution_time
    
    def _process(self): 
        pass

    def execute(self):
        print('\n' + self.__name + ' started')
        startTime = time()
        self._process()
        endTime = time()
        self.__executionTime = endTime - startTime
        print('Execution time:', self.getFormatedExecutionTime())
    
    def getFormatedExecutionTime(self):
        if self.__executionTime:
            hours, rem = divmod(self.__executionTime, 3600)
            minutes, seconds = divmod(rem, 60)
            return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)
        else:
            return 'unexecuted'
    
    def _getMemoryUsage(self):
        return self.__process.memory_info().rss