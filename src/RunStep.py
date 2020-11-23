from modules.Extraction import Extraction
from modules.PreProcessing import PreProcessing
from modules.TopicModeling import TopicModeling

steps = [Extraction(), PreProcessing(), TopicModeling()]

while True:
    print('\n')
    for i in range(len(steps)):
        print(i+1, '-', steps[i].stepName)
    print('0 - Exit\n')

    step = input('Choose a step to run: ')

    if step == '0':
        break
    elif step != '':
        for i in range(len(steps)):
            if str(i+1) == step:
                steps[i].execute()
                continue
        print('\nStep not found')