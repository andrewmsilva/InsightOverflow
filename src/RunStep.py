from modules.Extraction import Extraction
from modules.PreProcessing import PreProcessing
from modules.TopicModeling import TopicModeling
from modules.PostProcessing import PostProcessing

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('step', help='String containing the step name (Extraction, PreProcessing, TopicModeling or PostProcessing)', type=str)
args = parser.parse_args()

step = None
if args.step == 'Extraction':
    step = Extraction()
elif args.step == 'PreProcessing':
    step = PreProcessing()
elif args.step == 'TopicModeling':
    step = TopicModeling()
elif args.step == 'PostProcessing':
    step = PostProcessing()
else:
    raise parser.ArgumentTypeError('Step name must be Extraction, PreProcessing, TopicModeling or PostProcessing')

step.execute()