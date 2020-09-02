# Common imports
from smart_open import open
from time import time
from csv import DictReader

# Time counting
def start_process(process_name='Process'):
    print(process_name, 'started')
    return time()

def end_process(start_time):
    elapsed_time = time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('  Elapsed time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))

# Folders
data_folder = '../data/'
results_folder = '../results/'

# Data files
database_file = data_folder+'posts.xml'
posts_file = data_folder+'posts.csv'
clean_posts_file = data_folder+'clean-' + posts_file
enriched_posts_file = data_folder+'enriched-'  + posts_file
posts_header = ['date', 'author', 'content']

# Iterable of posts
def read_posts(csv_file, attribute=None, split=False):
    with open(csv_file, "r") as csv_file:
        for post in DictReader(csv_file):
            if attribute == None:
                yield post
            elif split == True:
                yield post[attribute].split()
            else:
                yield post[attribute]

# Results files
dictionary_file = data_folder+'dictionary.bin'
tfidf_file = data_folder+'tfidf-model.bin'
corpus_file = data_folder+'corpus.mm'
topic_model_file = data_folder+'topic-model.bin'