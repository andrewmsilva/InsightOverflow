from modules.Step import Step
from modules.Data import Posts, Users, Dates

from lxml import etree
from redis import Redis

class Extraction(Step):
    databaseFile = 'data/posts.xml'

    def __init__(self):
        super().__init__('Extraction')

    def _process(self):
        # Connect to Redis
        redis = Redis(host='localhost', port=6379, decode_responses=True)
        # Initialize data
        posts = Posts(overwrite=True)
        users = Users(overwrite=True)
        dates = Dates(overwrite=True)
        # Get posts
        total_count = 0
        for event, element in etree.iterparse(self.databaseFile, tag='row'):
            total_count += 1
            # Filter questions and answers
            post_type = int(element.get('PostTypeId'))
            if post_type != 1 and post_type != 2:
                continue
            # Get user and date
            user = element.get('OwnerUserId')
            date = element.get('CreationDate')[:10]
            if not user or not date:
                continue
            dates.append(date)
            users.append(user)
            # Get post
            post = element.get('Body')
            if post_type == 1:
                # Concatenate title and tags
                tags = element.get('Tags').replace('>', ' ').replace('<', '')
                index = element.get('Id')
                redis.set(index, tags)
                post += ' ' + element.get('Title') + ' ' + tags
            else:
                # Concatenate tags
                parent = element.get('ParentId')
                tags = redis.get(parent)
                if type(tags) == str:
                    post += ' ' + tags
            post = post.replace('\n', ' ').replace('\r', '')
            posts.append(post)
            # Clear memory
            element.clear()
            for ancestor in element.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]

        print('  Extracted:', len(posts))
        print('  Ignored:', total_count - len(posts))
        print('  Total:', total_count)