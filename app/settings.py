# Common imports
from time import time
from csv import DictReader

# Time counting
start_time = None
process_name = ""

def start(process=""):
    process_name = process
    start_time = time()
    print(process_name + ': started')

def end():
    execution_time = time() - start_time
    print(process_name + ': done in %0.4f'%execution_time)

# Folders
data_folder = '../data/'
results_folder = '../results/'

# Data files
posts_xml = 'posts.xml'
posts_csv = 'posts.csv'
clean_posts_csv = 'clean-' + posts_csv
enriched_posts_csv = 'enriched-'  + posts_csv
posts_header = ['date', 'author', 'body']

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
tfidf_bin = 'tfidf.bin'
topic_model_bin = 'topic-model.bin'