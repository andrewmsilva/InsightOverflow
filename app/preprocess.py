from time import time
from glob import glob

from csv import DictReader, DictWriter
from bs4 import BeautifulSoup

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# from nltk import download
# download('wordnet')
# download('averaged_perceptron_tagger')

from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet

start_time = time()

# Get all csv paths
data_path = '../data/'
all_files = glob(data_path+'*-posts.csv')

# Create function that get part of speech
def getPOS(token):
    tag = pos_tag([token])[0][1][0].upper()
    tag_dict = {
      "J": wordnet.ADJ,
      "N": wordnet.NOUN,
      "V": wordnet.VERB,
      "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

word_net = WordNetLemmatizer()
years = []

for csv_path in all_files:
    # Open CSVs
    with open(csv_path, "r") as csv_file:
        # Read posts
        for post in DictReader(csv_file):
            # Remove HTML tags
            soup = BeautifulSoup(post['content'], 'html.parser')
            for element in soup.find_all('code'):
                element.decompose()
            post['content'] = soup.get_text().replace('\n', ' ').replace('\r', '')
            # Preprocess post
            preprocessed = []
            for token in simple_preprocess(post['content'], deacc=True):
                if token not in STOPWORDS:
                    preprocessed.append(word_net.lemmatize(token, pos=getPOS(token)))
            # Write in CSV
            with open(data_path+'{}-preprocessed.csv'.format(post['year']), 'a') as f:
                writer = DictWriter(f, fieldnames=post.keys()) 
                if post['year'] not in years:
                    writer.writeheader()
                    years.append(post['year'])
                writer.writerow(post)

print('Preprocessing done after {:0.4f} seconds'.format(time() - start_time))