
class BaseStream(object):
    def __init__(self, textFile, maxLen=None, memory=True):
        self.__textFile = textFile
        self.__maxLen = maxLen
        self.__memory = memory
        self.__len = None
        self.__data = None
    
    def itemProcessing(self, data):
        return data
    
    def __loadData(self):
        self.__data = []
        with open(self.__textFile, "r") as f:
            self.__len = 0
            for row in f:
                if not self.__maxLen or self.__len < self.__maxLen:
                    self.__data.append(row)
                    self.__len += 1
                else:
                    break
    
    def __iterData(self):
        with open(self.__textFile, "r") as f:
            self.__len = 0
            for row in f:
                if not self.__maxLen or self.__len < self.__maxLen:
                    self.__len += 1
                    yield row
                else:
                    break
    
    def append(self, row):
        mode = "a"
        if not self.__len:
            mode = "w"
            self.__len = 0
            if self.__memory:
                self.__data = []
        
        with open(self.__textFile, mode, errors="surrogatepass") as f:
            f.write(str(row)+'\n')
            self.__len += 1
            if self.__memory:
                self.__data.append(row)

    def __iter__(self):
        if self.__memory:
            if not self.__data:
                self.__loadData()
            for item in self.__data:
                yield self.itemProcessing(item)
        else:
            for item in self.__iterData():
                yield self.itemProcessing(item)
    
    def __getitem__(self, key):
        if self.__memory:
            return self.itemProcessing(self.__data[key])
        else:
            tmpKey = 0
            for item in self.__iterData():
                if tmpKey == key:
                    return self.itemProcessing(item)
                else:
                    tmpKey += 1
    
    def __len__(self):
        if not self.__len:
            if self.__memory:
                self.__loadData()
            else:
                for _ in self.__iterData(): pass
        
        return self.__len
