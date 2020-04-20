from lxml import etree
from csv import DictWriter
from time import time

start_time = time()

data_path = '../data/posts.xml'
context = etree.iterparse(data_path, tag='row')

years = []
extracted_count = 0
total_count = 0

for event, element in context:
    total_count += 1
    attributes = element.keys()
    if 'CreationDate' in attributes and 'OwnerUserId' in attributes and 'Body' in attributes:
        # Get values
        extracted_count += 1
        post = {
            'id': extracted_count-1,
            'year': element.get('CreationDate')[:4],
            'author': element.get('OwnerUserId'),
            'content': element.get('Body')
        }
        if 'Title' in attributes:
            post['content'] += element.get('Title')
        # Write in CSV       
        with open('../data/{}-posts.csv'.format(post['year']), 'a') as f:
            writer = DictWriter(f, fieldnames=post.keys()) 
            if post['year'] not in years:
                writer.writeheader()
                years.append(post['year'])
            writer.writerow(post)
            f.close()
    # Clear memory
    element.clear()
    for ancestor in element.xpath('ancestor-or-self::*'):
        while ancestor.getprevious() is not None:
            del ancestor.getparent()[0]
del context

print('Data extraction done after {:0.4f} seconds'.format(time() - start_time))
print('  Extracted posts:', extracted_count)
print('  Unextracted posts:', total_count - extracted_count)
print('  Total posts:', total_count)