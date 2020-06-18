from csv import DictReader
from datetime import datetime

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import AuthorTopicModel

# Settings
path = '../data/'
source_file = 'preprocessed-posts.csv'

def getPosts():
    posts = []
    initial_date = None
    # Read questions and answers
    with open(path+source_file, "r") as csv_file:
        for post in DictReader(csv_file):
            # Create datetime object
            date = [ int(num) for num in post['date'].split('-') ]
            date = datetime(date[0], date[1], date[2])
            # Check if this is the first one
            if initial_date == None:
                initial_date = date
                posts.append(post)
            else:
                # Calculate differece in months
                months = (date.year - initial_date.year) * 12 + (date.month - initial_date.month)
                # If this post belongs to this quarter, append
                if months < 3:
                    posts.append(post)
                # If this post belongs to the next quarter, yield and start a new quarter
                else:
                    yield posts
                    initial_date = date
                    posts = [post]

def populateAuthors(posts):
    authors = {}
    for i in range(len(posts)):
        author = str(posts[i]['author'])
        if author not in list(authors.keys()):
            authors[author] = [i]
        else:
            authors[author].append(i)
    return authors

count = 0
for posts in getPosts():
    count += 1
    if count > 2: break

    # Get time window information
    initial_date = posts[0]['date'][:7]
    end_date = posts[-1]['date'][:7]
    print('\nTime window:', initial_date, 'to', end_date)

    # Populate authors
    authors = populateAuthors(posts)

    # Extract corpus
    corpus = [ post['body'].split() for post in posts ]
    del posts

    # Create dictionary
    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=150, no_above=0.5, keep_n=40000)

    # Create TF-IDF
    
    bow_corpus = [ dictionary.doc2bow(doc) for doc in corpus ]
    tfidf = TfidfModel(bow_corpus)
    tfidf_corpus = tfidf[bow_corpus].corpus

    # Create topic model
    model = AuthorTopicModel(
        tfidf_corpus,
        num_topics=8,
        id2word=dictionary,
        author2doc=authors,
        random_state=0
    )

    # Print resuts
    print('  Topics')
    for idx, topic in model.print_topics(-1):
        print('    {}: {}'.format(idx, topic))

    # print('  Authors')
    # for author in model.id2author.values():
    #     print('    {}: {}'.format(author, model.get_author_topics(author)))