
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('step', help='String containing the step name (Extraction, PreProcessing, TopicModeling or PostProcessing)', type=str)
args = parser.parse_args()

step = None
if args.step == 'Extraction':
    from modules.Extraction import Extraction
    step = Extraction()
elif args.step == 'PreProcessing':
    from modules.PreProcessing import PreProcessing
    step = PreProcessing()
elif args.step == 'TopicModeling':
    from modules.TopicModeling import TopicModeling
    step = TopicModeling()
elif args.step == 'PostProcessing':
    from modules.PostProcessing import PostProcessing
    step = PostProcessing()
else:
    raise parser.ArgumentTypeError('Step name must be Extraction, PreProcessing, TopicModeling or PostProcessing')

step.execute()