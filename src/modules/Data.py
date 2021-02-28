
class Stream(object):
    def __init__(self, textFile, maxLen=None):
        self.__textFile = textFile
        self.__maxLen = maxLen
        self.__len = None
        self.__data = None
    
    def __loadData(self):
        self.__data = []
        with open(self.__textFile, "r") as f:
            self.__len = 0
            for row in f:
                if not self.__maxLen or self.__maxLen < self.__len:
                    self.__data.append(row)
                    self.__len += 1
    
    def append(self, row):
        mode = "a"
        if not self.__data:
            mode = "w"
            self.__data = []
            self.__len = 0
        
        with open(self.__textFile, mode, errors="surrogatepass") as f:
            f.write(str(row)+'\n')
            self.__data.append(row)
            self.__len += 1

    def __iter__(self):
        if not self.__data:
            self.__loadData()
        
        for item in data:
            yield data
    
    def __getitem__(self, key):
        return self.__data[key]
    
    def __len__(self):
        if not self.__len:
            self.__loadData()
        return self.__len

class Posts(object):

    def __init__(self, preProcessed=False, maxLen=None):
        self.users = Stream("data/users.txt", maxLen)
        self.dates = Stream("data/dates.txt", maxLen)

        if preProcessed:
            self.contents = Stream("data/data/pre-processed-contents.txt", maxLen)
        else:
            self.contents = Stream("data/contents.txt", maxLen)
    
    def __iter__(self):
        for (content, user, date) in zip(self.__contents, self.__users, self.__dates):
            yield {
                'content': content,
                'user': user,
                'date': date,
            }