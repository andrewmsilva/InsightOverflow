# Common imports
from time import time
from csv import DictReader

# Folders
data_folder = '../data/'
results_folder = '../results/'

# Files
posts_xml = 'posts.xml'
posts_csv = 'posts.csv'
clean_posts_csv = 'clean-' + posts_csv
enriched_post_csv = 'enriched-'  + posts_csv

posts_header = ['date', 'author', 'body']

# Itarable of posts
def read_posts(csv_file, attribute=None, split=False):
    with open(data_folder+csv_file, "r") as csv_file:
        for post in DictReader(csv_file):
            if attribute == None:
                yield post
            elif split == True:
                yield post[attribute].split()
            else:
                yield post[attribute]