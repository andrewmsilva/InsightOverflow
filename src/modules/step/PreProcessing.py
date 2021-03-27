from .BaseStep import BaseStep
from ..data.Posts import Posts

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

class PreProcessing(BaseStep):

    def __init__(self):
        super().__init__('Pre-processing')
        self.__wordNet = WordNetLemmatizer()
        self.__posts = Posts(memory=False)
        self.__pPosts = Posts(preProcessed=True, memory=False)
    
    def __clearHTML(self, content):
        soup = BeautifulSoup(content, 'lxml')
        for tag in soup.find_all('code'):
            tag.decompose()
        return soup.get_text()
    
    def __getPOS(self, token):
        tag = pos_tag([token])[0][1][0].upper()
        tagDict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tagDict.get(tag, wordnet.NOUN)
    
    def __applyNLP(self, content):
        doc = []
        for token in simple_preprocess(content, deacc=True):
            token = self.__wordNet.lemmatize(token, pos=self.__getPOS(token))
            if token not in STOPWORDS and len(token) > 1:
                doc.append(token)
        return doc
    
    def __cleaning(self):
        cleanContents = []
        for content in self.__posts.contents:
            content = self.__clearHTML(content)
            content = self.__applyNLP(content)
            cleanContents.append(content)
        return cleanContents
    
    def __makeBigrams(self, cleanContents):
        # Train Phrases model
        bigramModel = Phrases(cleanContents)

        # Make bi-grams
        for content in cleanContents:
            content = ' '.join(bigramModel[content])
            self.__pPosts.contents.append(content)

    def _process(self):
        self.__makeBigrams(self.__cleaning())