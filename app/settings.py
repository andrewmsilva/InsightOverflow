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
posts_xml = 'posts.xml'
posts_csv = 'posts.csv'
clean_posts_csv = 'clean-' + posts_csv
enriched_posts_csv = 'enriched-'  + posts_csv
posts_header = ['date', 'author', 'content']

# Iterable of posts
def read_posts(csv_file, attribute=None, split=False):
    with open(data_folder+csv_file, "r") as csv_file:
        for post in DictReader(csv_file):
            if attribute == None:
                yield post
            elif split == True:
                yield post[attribute].split()
            else:
                yield post[attribute]

# Results files
dictionary_bin = 'dictionary.bin'
tfidf_bin = 'tfidf-model.bin'
corpus_mm = 'corpus.mm'
topic_model_bin = 'topic-model.bin'