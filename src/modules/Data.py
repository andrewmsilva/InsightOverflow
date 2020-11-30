from smart_open import open

class Stream(object):

    def __init__(self, output_file, overwrite=False, splitted=False):
        self.__length = None
        self.__outputFile = output_file
        self.__overwrite = overwrite
        self.__splitted = splitted

    def __iter__(self):
        with open(self.__outputFile, "r") as txt_file:
            for data in txt_file:
                try:
                    yield int(data)
                except:
                    if self.__splitted:
                        yield data.split()
                    else:
                        yield data
    
    def __len__(self):
        if not self.__length:
            self.__length = 0  
            for data in self:
                self.__length += 1
        return self.__length
    
    def append(self, data):
        mode = 'a'
        if self.__overwrite:
            self.__overwrite = False
            self.__length = 0
            mode = 'w'

        with open(self.__outputFile, mode, errors='surrogatepass') as txt_file:
            txt_file.write(str(data)+'\n')
            self.__length += 1

class Users(Stream):
    def __init__(self, overwrite=False, splitted=False):
        super().__init__('data/users.txt', overwrite, splitted)

class Dates(Stream):
    def __init__(self, overwrite=False, splitted=False):
        super().__init__('data/dates.txt', overwrite, splitted)

class Contents(Stream):
    def __init__(self, overwrite=False, splitted=False):
        super().__init__('data/contents.txt', overwrite, splitted)

class PreProcessedContents(Stream):
    def __init__(self, overwrite=False, splitted=False):
        super().__init__('data/pre-processed-contents.txt', overwrite, splitted)