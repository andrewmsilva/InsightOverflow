from lxml import etree
from csv import DictWriter
from time import time
from redis import Redis

print('Extraction started')
start_time = time()

# Settings
path = '../data/'
source_file = 'posts.xml'
results_file = 'posts.csv'
columns = ['id', 'date', 'author', 'body']

# Connect to Redis
redis = Redis(host='localhost', port=6379)

# Create CSV
with open(path+results_file, 'w', errors='surrogatepass') as f:
    writer = DictWriter(f, fieldnames=columns) 
    writer.writeheader()

# Get posts
extracted_count= 0
total_count = 0
for event, element in etree.iterparse(path+source_file, tag='row'):
    total_count += 1
    # Filter questions and answers
    post_type = element.get('PostTypeId')
    score = int(element.get('Score'))
    if (post_type == '1' or post_type == '2') and score > 0:
        # Get values
        extracted_count += 1
        post = {
            'id': element.get('Id'),
            'date': element.get('CreationDate')[:10],
            'author': element.get('OwnerUserId'),
            'body': element.get('Body')
        }
        # If it is a question, append title and tags
        if post_type == '1':
            tags = element.get('Tags')
            post['body'] += ' ' + element.get('Title') + ' ' + tags
            redis.set(post['id'], tags)
        # If it is an answer, append parent tags
        else:
            parent = element.get('ParentId')
            tags = redis.get(parent)
            post['body'] += ' ' + str(tags)
        # Write in CSV
        with open(path+results_file, 'a', errors='surrogatepass') as f:
            writer = DictWriter(f, fieldnames=columns) 
            writer.writerow(post)
    # Clear memory
    element.clear()
    for ancestor in element.xpath('ancestor-or-self::*'):
        while ancestor.getprevious() is not None:
            del ancestor.getparent()[0]

print('  Extracted:', extracted_count)
print('  Ignored:', total_count - extracted_count)
print('  Total:', total_count)
print('Done in {:0.4f} seconds'.format(time() - start_time))

# Data extraction done after 22951.1279 seconds