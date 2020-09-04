from time import time

class Step(object):
    stepName = 'Unamed step'
    executionTime = None

    def __init__(self, stepName=None):
        if type(stepName) == str:
            self.stepName = stepName
    
    def _process(self): 
        pass

    def execute(self):
        print('\n' + self.stepName + ' started')
        startTime = time()
        self._process()
        endTime = time()
        self.executionTime = endTime - startTime
        print('Execution time:', self.getFormatedExecutionTime())
    
    def getFormatedExecutionTime(self):
        if self.executionTime:
            hours, rem = divmod(self.executionTime, 3600)
            minutes, seconds = divmod(rem, 60)
            return '{:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds)
        else:
            return 'unexecuted'