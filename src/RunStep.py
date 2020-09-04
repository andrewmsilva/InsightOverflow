from modules.Extraction import Extraction

while True:
    print('\n1 - Extraction')
    print('2 - Pre-processing')
    print('3 - Topic modeling')
    print('0 - Exit\n')
    step = input('Choose a step to run: ')

    if step == '0':
        break
    elif step == '1':
        step = Extraction()
        step.execute()
    else:
        print('\nStep not found')