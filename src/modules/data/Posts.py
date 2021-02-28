from .BaseStream import BaseStream

class Posts(object):

    def __init__(self, preProcessed=False, maxLen=None):
        self.users = BaseStream("data/users.txt", maxLen)
        self.dates = BaseStream("data/dates.txt", maxLen)

        if preProcessed:
            self.contents = BaseStream("data/data/pre-processed-contents.txt", maxLen)
        else:
            self.contents = BaseStream("data/contents.txt", maxLen)
    
    def __iter__(self):
        for (content, user, date) in zip(self.__contents, self.__users, self.__dates):
            yield {
                'content': content,
                'user': user,
                'date': date,
            }