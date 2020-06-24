from settings import *

from csv import DictReader, DictWriter
from bs4 import BeautifulSoup

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# from nltk import download
# download('wordnet')
# download('averaged_perceptron_tagger')

from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet

print('Cleaning started')
start_time = time()

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

# Create CSV
with open(data_path+clean_posts_csv, 'w', errors='surrogatepass') as result_file:
    writer = DictWriter(result_file, fieldnames=posts_header) 
    writer.writeheader()

# Clean posts
word_net = WordNetLemmatizer()
with open(data_path+posts_csv, "r") as csv_file:
    for post in DictReader(csv_file):
        # Remove HTML tags
        soup = BeautifulSoup(post['body'], 'lxml')
        for tag in soup.find_all('code'):
            tag.decompose()
        post['body'] = soup.get_text().replace('\n', ' ').replace('\r', '')
        # Tokenize, remove stopwords and lemmatize
        post['body'] = [ word_net.lemmatize(token, pos=getPOS(token)) for token in simple_preprocess(post['body'], deacc=True) if token not in STOPWORDS ]
        post['body'] = ' '.join(post['body'])
        # Write in CSV
        with open(data_path+clean_posts_csv, 'a', errors='surrogatepass') as result_file:
            writer = DictWriter(result_file, fieldnames=posts_header) 
            writer.writerow(post)

print('Done in {:0.4f} seconds'.format(time() - start_time))