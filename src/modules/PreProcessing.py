from modules.Step import Step
from modules.Data import Stream, Contents, PreProcessedContents

from bs4 import BeautifulSoup
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

# from nltk import download
# download('wordnet')
# download('averaged_perceptron_tagger')

from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from os import remove

class PreProcessing(Step):
    __tempFile = 'data/clean-contents.txt'

    def __init__(self):
        super().__init__('Pre-processing')
    
    def __getPOS(self, token):
        tag = pos_tag([token])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def __cleaning(self):
        word_net = WordNetLemmatizer()
        contents = Contents()
        clean_contents = Stream(self.__tempFile, overwrite=True)
        for content in contents:
            # Remove HTML tags
            soup = BeautifulSoup(content, 'lxml')
            for tag in soup.find_all('code'):
                tag.decompose()
            content = soup.get_text()
            # Tokenize, remove stopwords and lemmatize
            content = [ word_net.lemmatize(token, pos=self.__getPOS(token)) for token in simple_preprocess(content, deacc=True) if token not in STOPWORDS ]
            content = ' '.join(content)
            clean_contents.append(content)

    def __enrichment(self):
        # Train Phrases model
        clean_contents = Stream(self.__tempFile)
        bigram_model = Phrases(clean_contents, min_count=1)

        # Make bi-grams
        pre_processed_contents = PreProcessedContents(overwrite=True)
        for content in clean_contents:
            content = ' '.join(bigram_model[content.split()])
            pre_processed_contents.append(content)
        # Remove temporary file
        remove(self.__tempFile)

    def _process(self):
        self.__cleaning()
        self.__enrichment()