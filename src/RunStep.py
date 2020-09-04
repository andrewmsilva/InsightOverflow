from modules.Extraction import Extraction
from modules.PreProcessing import PreProcessing

steps = [Extraction(), PreProcessing()]

while True:
    print('\n')
    for i in range(len(steps)):
        print(i+1, '-', steps[i].stepName)
    print('0 - Exit\n')

    step = input('Choose a step to run: ')

    if step == '0':
        break
    else:
        for i in range(len(steps)):
            if str(i+1) == step:
                steps[i].execute()
                continue
        print('\nStep not found')