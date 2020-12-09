from modules.Data import PreProcessedContents

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

class Corpus(object):

    def __init__(self):
        # Load collection
        self.__contents = PreProcessedContents(splitted=True)
        # Build dictionary
        self.__dictionaryFile = 'results/dictionary.bin'
        try:
            self.__dictionary = Dictionary.load(self.__dictionaryFile)
        except:
            self.__dictionary = Dictionary(self.__contents)
            self.__dictionary.filter_extremes(no_below=int(0.05*len(self.__contents)), no_above=0.8, keep_n=100000)
            self.__dictionary.save(self.__dictionaryFile)
        # Build IF-IDF
        self.__tfidfFile = 'results/tfidf.bin'
        try:
            self.__tfidf = TfidfModel.load(self.__tfidfFile)
        except:
            self.__tfidf = TfidfModel(
                self.__bow(),
                id2word=self.__dictionary
            )
            self.__tfidf.save(self.__tfidfFile)

    def getDictionary(self):
        return self.__dictionary
    
    def getContents(self):
        return self.__contents
    
    def __bow(self):
        for document in self.__contents:
            yield self.__dictionary.doc2bow(document)
        
    def __iter__(self):
        for document in self.__bow():
            yield self.__tfidf[document]

    def __len__(self):
        return len(self.__contents)