from settings import *

from csv import DictWriter
from bs4 import BeautifulSoup

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# from nltk import download
# download('wordnet')
# download('averaged_perceptron_tagger')

from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet

start_time = start_process('Cleaning')

# Create function that get part of speech
def get_pos(token):
    tag = pos_tag([token])[0][1][0].upper()
    tag_dict = {
      "J": wordnet.ADJ,
      "N": wordnet.NOUN,
      "V": wordnet.VERB,
      "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

# Create CSV
with open(data_folder+clean_posts_file, 'w', errors='surrogatepass') as result_file:
    writer = DictWriter(result_file, fieldnames=posts_header) 
    writer.writeheader()

# Clean posts
word_net = WordNetLemmatizer()
for post in read_posts(posts_file):
    # Remove HTML tags
    soup = BeautifulSoup(post['content'], 'lxml')
    for tag in soup.find_all('code'):
        tag.decompose()
    post['content'] = soup.get_text().replace('\n', ' ').replace('\r', '')
    # Tokenize, remove stopwords and lemmatize
    post['content'] = [ word_net.lemmatize(token, pos=get_pos(token)) for token in simple_preprocess(post['content'], deacc=True) if token not in STOPWORDS ]
    post['content'] = ' '.join(post['content'])
    # Write in CSV
    with open(data_folder+clean_posts_file, 'a', errors='surrogatepass') as result_file:
        writer = DictWriter(result_file, fieldnames=posts_header) 
        writer.writerow(post)

end_process(start_time)