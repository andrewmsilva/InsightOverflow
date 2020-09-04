from smart_open import open

class Data(object):
    __length = None

    def __init__(self, output_file, overwrite):
        self.outputFile = output_file
        self.overwrite = overwrite

    def __iter__(self):
        with open(self.outputFile, "r") as csv_file:
            for data in csv.reader(csv_file):
                yield data[0]
    
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

class Users(Data):
    def __init__(self, overwrite=False):
        super().__init__('data/users.txt', overwrite)

class Dates(Data):
    def __init__(self, overwrite=False):
        super().__init__('data/dates.txt', overwrite)

class Posts(Data):
    def __init__(self, overwrite=False):
        super().__init__('data/posts.txt', overwrite)

class PreProcessedPosts(Data):
    def __init__(self, overwrite=False):
        super().__init__('data/pre-processed-posts.txt', overwrite)