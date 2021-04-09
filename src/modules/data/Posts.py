from .BaseStream import BaseStream

class Posts(object):

    def __init__(self, preProcessed=False, maxLen=None, memory=True, splitted=False):
        self.users = BaseStream("data/users.txt", maxLen, memory)
        self.dates = BaseStream("data/dates.txt", maxLen, memory)

        if preProcessed:
            self.contents = BaseStream("data/pre-processed-contents.txt", maxLen, memory)
        else:
            self.contents = BaseStream("data/contents.txt", maxLen, memory)
        
        if splitted:
            self.contents.setItemProcessing(self.__split)
    
    def __iter__(self):
        for (content, user, date) in zip(self.contents, self.users, self.dates):
            yield {
                'content': content,
                'user': user.replace('\n', ''),
                'date': date.replace('\n', ''),
            }
    
    def __len__(self):
        return len(self.contents)
    
    def __split(self, content):
        return content.split()