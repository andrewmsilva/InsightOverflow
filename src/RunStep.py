from modules.Extraction import Extraction
from modules.PreProcessing import PreProcessing
from modules.TopicModeling import TopicModeling
from modules.PostProcessing import PostProcessing

steps = [Extraction(), PreProcessing(), TopicModeling(), PostProcessing()]

while True:
    print('\n')
    for i in range(len(steps)):
        print(i+1, '-', steps[i].getName())
    print('0 - Exit\n')

    step = input('Choose a step to run: ')

    if step == '0':
        break
    elif step != '':
        step = int(step) - 1
        if step in range(len(steps)):
            steps[step].execute()
        else:
            print('\nStep not found')