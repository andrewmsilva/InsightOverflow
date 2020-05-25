from lxml import etree
from csv import DictWriter
from time import time

start_time = time()

# Create XML context
data_path = '../data/'
context = etree.iterparse(data_path+'posts.xml', tag='row')

# Settings
years = []
extracted_count = 0
total_count = 0

# Get posts
for event, element in context:
    total_count += 1
    # Filter posts
    post_type = element.get('PostTypeId')
    if 'OwnerUserId' in element.keys() and (post_type is '1' or post_type is '2'):
        # Get values
        extracted_count += 1
        post = {
            'id': extracted_count-1,
            'year': element.get('CreationDate')[:4],
            'author': element.get('OwnerUserId'),
            'content': element.get('Body')
        }
        # Append title
        if 'Title' in element.keys():
            post['content'] += element.get('Title')
        # Write in CSV
        with open(data_path+'{}-posts.csv'.format(post['year']), 'a', errors='surrogatepass') as f:
            writer = DictWriter(f, fieldnames=post.keys()) 
            if post['year'] not in years:
                writer.writeheader()
                years.append(post['year'])
            writer.writerow(post)
    # Clear memory
    element.clear()
    for ancestor in element.xpath('ancestor-or-self::*'):
        while ancestor.getprevious() is not None:
            del ancestor.getparent()[0]
del context

print('Data extraction done after {:0.4f} seconds'.format(time() - start_time))
print('  Extracted:', extracted_count)
print('  Ignored:', total_count - extracted_count)
print('  Total:', total_count)