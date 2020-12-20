from modules.Step import Step
from modules.Data import Stream, Contents, PreProcessedContents

from bs4 import BeautifulSoup
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import re

# from nltk import download
# download('wordnet')
# download('averaged_perceptron_tagger')

from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from os import remove

class PreProcessing(Step):

    def __init__(self):
        super().__init__('Pre-processing')
        self.__tempFile = 'data/clean-contents.txt'
        self.__wordNet = WordNetLemmatizer()
    
    def __clearHTML(self, content):
        soup = BeautifulSoup(content, 'lxml')
        for tag in soup.find_all('code'):
            tag.decompose()
        return soup.get_text()
    
    def __getPOS(self, token):
        tag = pos_tag([token])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def __applyNLP(self, content):
        doc = []
        for token in simple_preprocess(content, deacc=True):
            token = self.__wordNet.lemmatize(token, pos=self.__getPOS(token))
            if token not in STOPWORDS and len(token) > 1:
                doc.append(token)
        return doc
    
    def __cleaning(self):
        contents = Contents()
        clean_contents = Stream(self.__tempFile, overwrite=True)
        for content in contents:
            content = self.__clearHTML(content)
            content = self.__applyNLP(content)
            clean_contents.append(" ".join(content))
            yield content

    def _process(self):
        # Train Phrases model
        bigram_model = Phrases(self.__cleaning())

        # Make bi-grams
        clean_contents = Stream(self.__tempFile, splitted=True)
        pre_processed_contents = PreProcessedContents(overwrite=True)
        for content in clean_contents:
            content = ' '.join(bigram_model[content])
            pre_processed_contents.append(content)
        # Remove temporary file
        remove(self.__tempFile)