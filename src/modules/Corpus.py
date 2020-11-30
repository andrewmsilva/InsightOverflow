from modules.Data import PreProcessedContents

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

class Corpus(object):

    def __init__(self):
        self.__contents = PreProcessedContents(splitted=True)

        self.__dictionary = None
        self.__dictionaryFile = 'results/dictionary.bin'

        self.__tfidf = None
        self.__tfidfFile = 'results/tfidf.bin'

        self.__len = None
    
    def getDictionary(self):
        return self.__dictionary
    
    def getContents(self):
        return self.__contents
    
    def build(self, no_below=None, no_above=None, keep_n=None):
        self.__dictionary = Dictionary(self.__contents)

        if no_below != None and no_above != None and keep_n != None:
            self.__dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        
        self.__tfidf = TfidfModel(
            self.__bow(),
            id2word=self.__dictionary
        )
    
    def __bow(self):
        count = None
        if not self.__len:
            count = 0
        
        for document in self.__contents:
            if isinstance(count, int):
                count += 1
            yield self.__dictionary.doc2bow(document)
        
        if isinstance(count, int):
            self.__len = count
        
    def __iter__(self):
        for document in self.__bow():
            yield self.__tfidf[document]

    def __len__(self):
        if not self.__len:
            self.__len = len(self.__contents)
        return self.__len
    
    def save(self):
        self.__dictionary.save(self.__dictionaryFile)
        self.__tfidf.save(self.__tfidfFile)
    
    def load(self):
        self.__dictionary = Dictionary.load(self.__dictionaryFile)
        self.__tfidf = TfidfModel.load(self.__tfidfFile)