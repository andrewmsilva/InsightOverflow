from settings import *

from csv import DictReader

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary

print('Enrichment started')
start_time = time()

# Train Phrases model
bigram_model = Phrases(read_posts(clean_posts_csv, attribute='body', split=True), min_count=1)

# create CSV
with open(data_folder+clean_posts_csv, 'w', errors='surrogatepass') as result_file:
    writer = DictWriter(result_file, fieldnames=posts_header) 
    writer.writeheader()

# Make bi-grams
for post in read_posts(clean_posts_csv):
    # Concatenate bi-grams
    post['body'] = post['body'].split()
    bigrams = [ bigram for bigram in bigram_model[post['body']] if '_' in bigram ]
    post['body'] += bigrams
    # Add post to dictionary
    dictionary.add_documents([post['body']])
    # Write in CSV
    post['body'] = ' '.join(post['body'])
    with open(data_folder+enriched_post_csv, 'a', errors='surrogatepass') as csv_file:
        writer = DictWriter(result_file, fieldnames=posts_header) 
        writer.writerow(post)

print('Done in {:0.4f} seconds'.format(time() - start_time))