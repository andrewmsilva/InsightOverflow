from .BaseStream import BaseStream

class Posts(object):

    def __init__(self, preProcessed=False, maxLen=None, memory=True):
        self.users = BaseStream("data/users.txt", maxLen, memory)
        self.dates = BaseStream("data/dates.txt", maxLen, memory)

        if preProcessed:
            self.contents = BaseStream("data/pre-processed-contents.txt", maxLen, memory)
        else:
            self.contents = BaseStream("data/contents.txt", maxLen, memory)
    
    def __iter__(self):
        for (content, user, date) in zip(self.__contents, self.__users, self.__dates):
            yield {
                'content': content,
                'user': user,
                'date': date,
            }
    
    def __len__(self):
        return len(self.contents)