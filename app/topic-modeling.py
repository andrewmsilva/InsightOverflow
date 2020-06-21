from csv import DictReader
from datetime import datetime

from gensim.corpora import Dictionary
from gensim.models import Phrases, TfidfModel, AuthorTopicModel
from gensim.models.wrappers import LdaMallet
from gensim.models.phrases import Phraser
from gensim.models import CoherenceModel

# Settings
path = '../data/'
source_file = 'preprocessed-posts.csv'
mallet_path = 'mallet-2.0.8/bin/mallet'

def getPosts(k_months):
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
                # If this post belongs to this set of months, append
                if months < k_months:
                    posts.append(post)
                # If this post belongs to the next set of months, yield and start a new set of months
                else:
                    yield posts
                    initial_date = date
                    posts = [post]

count = 0
for posts in getPosts(6):
    count += 1
    if count > 2: break

    # Get time window information
    initial_date = posts[0]['date'][:7]
    end_date = posts[-1]['date'][:7]
    print('\nTime window:', initial_date, 'to', end_date)

    # Populate authors
    authors = {}
    for i in range(len(posts)):
        author = str(posts[i]['author'])
        if author not in list(authors.keys()):
            authors[author] = [i]
        else:
            authors[author].append(i)

    # Make bigrams
    posts = [ post['body'].split() for post in posts ]
    bigram_model = Phraser(Phrases(posts)) # Higher threshold, fewer phrases min_count=5, threshold=100
    corpus = [ bigram_model[post] for post in posts ]

    # Create dictionary
    dictionary = Dictionary(corpus)
    # dictionary.filter_extremes(no_below=40, no_above=0.8, keep_n=4000)

    # Create TF-IDF
    corpus = [ dictionary.doc2bow(doc) for doc in corpus ]
    tfidf = TfidfModel(corpus)
    corpus = tfidf[corpus].corpus

    # Create topic model
    # topic_model = LdaMallet(
    #     mallet_path,
    #     corpus=corpus,
    #     id2word=dictionary,
    #     num_topics=10,
    #     random_seed=10
    # )
    
    topic_model = AuthorTopicModel(
        corpus,
        num_topics=10,
        id2word=dictionary,
        author2doc=authors,
        iterations=200,
        random_state=10
    )

    # Print resuts
    print('  Topics')
    for idx, topic in topic_model.print_topics(-1):
        print('    {}: {}'.format(idx, topic))

    # print('  Authors')
    # for author in topic_model.id2author.values():
    #     print('    {}: {}'.format(author, topic_model.get_author_topics(author)))