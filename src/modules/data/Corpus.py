from .BaseStream import BaseStream
from .Posts import Posts

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

def split(string):
    return string.split()

class Corpus(object):

    def __init__(self):
        self.__posts = Posts(preProcessed=True)
        self.__posts.contents.itemProcessing = split
        self.__corpus = None
        
        self.__dictionaryFile = 'results/dictionary.bin'
        self.__tfidfFile = 'results/tfidf.bin'
    
    def __buildDictionary(self):
        try:
            self.__dictionary = Dictionary.load(self.__dictionaryFile)
        except:
            self.__dictionary = Dictionary(self.__posts.contents)
            self.__dictionary.filter_extremes(no_below=int(0.05*len(self.__posts.contents)), no_above=0.8, keep_n=10000)
            self.__dictionary.save(self.__dictionaryFile)
    
    def __buildTFIDF(self):
        try:
            self.__tfidf = TfidfModel.load(self.__tfidfFile)
        except:
            self.__tfidf = TfidfModel(
                self.__bow(),
                id2word=self.__dictionary
            )
            self.__tfidf.save(self.__tfidfFile)
    
    def __bow(self):
        for content in self.__posts.contents:
            yield self.__dictionary.doc2bow(content)
    
    def build(self):
        self.__buildDictionary()
        self.__buildTFIDF()

        self.__corpus = []
        for content in self.__bow():
            self.__corpus.append(self.__tfidf[content])

    def getDictionary(self):
        return self.__dictionary
    
    def getTexts(self):
        return self.__posts.contents
    
    def __iter__(self):
        if not self.__corpus:
            self.build()
        for content in self.__corpus:
            yield content

    def __getitem__(self, key):
        return self.__corpus[key]

    def __len__(self):
        return len(self.__posts)