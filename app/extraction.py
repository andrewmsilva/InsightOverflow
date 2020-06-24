from settings import *

from lxml import etree
from csv import DictWriter
from time import time
from redis import Redis

print('Extraction started')
start_time = time()

# Connect to Redis
redis = Redis(host='localhost', port=6379, decode_responses=True)

# Create CSV
with open(data_folder+posts_csv, 'w', errors='surrogatepass') as f:
    writer = DictWriter(f, fieldnames=posts_header) 
    writer.writeheader()

# Get posts
extracted_count = 0
total_count = 0
for event, element in etree.iterparse(data_folder+posts_xml, tag='row'):
    total_count += 1
    # Filter questions and answers
    post_type = int(element.get('PostTypeId'))
    if post_type == 1 or post_type == 2:
        # Get and save tags
        tags = ''
        if post_type == 1:
            index = element.get('Id')
            tags = element.get('Tags').replace('>', ' ').replace('<', '')
            redis.set(index, tags)
        else:
            parent = element.get('ParentId')
            tags = redis.get(parent)
            if type(tags) != str:
                tags = ''
        # Filter score
        score = int(element.get('Score'))
        if score > 0:
            # Get title
            title = ''
            if post_type == 1:
                title = element.get('Title')
            # Get other information
            post = {
                'date': element.get('CreationDate')[:10],
                'author': element.get('OwnerUserId'),
                'body': element.get('Body') + ' ' + title + ' ' + tags
            }
            # Write in CSV
            with open(data_folder+posts_csv, 'a', errors='surrogatepass') as f:
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
print('Done in {:0.4f} seconds'.format(time() - start_time))