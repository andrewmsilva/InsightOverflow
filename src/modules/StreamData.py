from smart_open import open

class StreamData(object):
    __length = None

    def __init__(self, output_file, overwrite=False):
        self.outputFile = output_file
        self.overwrite = overwrite

    def __iter__(self):
        with open(self.outputFile, "r") as txt_file:
            for data in txt_file:
                try:
                    yield int(data)
                except:
                    yield data
    
    def __len__(self):
        if not self.__length:
            self.__length = 0  
            for data in self:
                self.__length += 1
        return self.__length
    
    def append(self, data):
        mode = 'a'
        if self.overwrite:
            self.overwrite = False
            self.__length = 0
            mode = 'w'

        with open(self.outputFile, mode, errors='surrogatepass') as txt_file:
            txt_file.write(str(data)+'\n')
            self.__length += 1

class Users(StreamData):
    def __init__(self, overwrite=False):
        super().__init__('data/users.txt', overwrite)

class Dates(StreamData):
    def __init__(self, overwrite=False):
        super().__init__('data/dates.txt', overwrite)

class Contents(StreamData):
    def __init__(self, overwrite=False):
        super().__init__('data/contents.txt', overwrite)

class PreProcessedContents(StreamData):
    def __init__(self, overwrite=False):
        super().__init__('data/pre-processed-contents.txt', overwrite)