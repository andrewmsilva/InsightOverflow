from settings import *

from lxml import etree
from csv import DictWriter
from redis import Redis

start_time = start_process('Extraction')

# Connect to Redis
redis = Redis(host='localhost', port=6379, decode_responses=True)

# Create CSV
with open(data_folder+posts_file, 'w', errors='surrogatepass') as f:
    writer = DictWriter(f, fieldnames=posts_header) 
    writer.writeheader()

# Get posts
extracted_count = 0
total_count = 0
for event, element in etree.iterparse(data_folder+database_file, tag='row'):
    total_count += 1
    # Filter questions and answers
    post_type = int(element.get('PostTypeId'))
    if post_type == 1 or post_type == 2:
        # Get title and save tags
        title = ''
        tags = ''
        if post_type == 1:
            index = element.get('Id')
            title = element.get('Title')
            tags = element.get('Tags').replace('>', ' ').replace('<', '')
            redis.set(index, tags)
        else:
            parent = element.get('ParentId')
            tags = redis.get(parent)
            if type(tags) != str:
                tags = ''
        # Get other information
        post = {
            'date': element.get('CreationDate')[:10],
            'author': element.get('OwnerUserId'),
            'content': element.get('Body') + ' ' + title + ' ' + tags
        }
        # Write in CSV
        with open(data_folder+posts_file, 'a', errors='surrogatepass') as f:
            writer = DictWriter(f, fieldnames=posts_header) 
            writer.writerow(post)
        extracted_count += 1
    # Clear memory
    element.clear()
    for ancestor in element.xpath('ancestor-or-self::*'):
        while ancestor.getprevious() is not None:
            del ancestor.getparent()[0]

print('  Extracted:', extracted_count)
print('  Ignored:', total_count - extracted_count)
print('  Total:', total_count)

end_process(start_time)