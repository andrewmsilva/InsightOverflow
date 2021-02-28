from modules.Step import Step
from modules.Data import Posts

from lxml import etree
from redis import Redis

class Extraction(Step):

    def __init__(self):
        super().__init__('Extraction')
        self.databaseFile = 'data/posts.xml'

    def _process(self):
        # Connect to Redis
        redis = Redis(host='localhost', port=6379, decode_responses=True)
        posts = Posts()
        # Get posts
        total_count = 0
        for event, element in etree.iterparse(self.databaseFile, tag='row'):
            total_count += 1
            # Filter questions and answers
            post_type = int(element.get('PostTypeId'))
            if post_type != 1 and post_type != 2: continue
            # Get user and date
            user = element.get('OwnerUserId')
            date = element.get('CreationDate')[:10]
            if not user or not date: continue
            # Get content
            content = element.get('Body')
            if not content: continue
            if post_type == 1:
                # Concatenate title and tags
                tags = element.get('Tags').replace('>', ' ').replace('<', '')
                index = element.get('Id')
                redis.set('post'+index, tags)
                content += ' ' + element.get('Title') + ' ' + tags
            else:
                # Concatenate tags
                parent = element.get('ParentId')
                tags = redis.get('post'+parent)
                if type(tags) == str:
                    content += ' ' + tags
            content = content.replace('\n', ' ').replace('\r', '')
            content = content.encode("ascii", errors="ignore").decode()
            # Save data
            posts.contents.append(content)
            posts.dates.append(date)
            posts.users.append(user)
            # Clear memory
            element.clear()
            for ancestor in element.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]

        print('  Extracted:', len(contents))
        print('  Ignored:', total_count - len(contents))
        print('  Total:', total_count)