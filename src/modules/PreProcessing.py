from modules.Step import Step
from modules.Data import Data, Posts, PreProcessedPosts

from os import remove

class PreProcessing(Step):
    __tempFile = 'data/clean-posts.csv'

    def __init__(self):
        super().__init__('Pre-processing')
    
    def __getPOS(self, token):
        tag = pos_tag([token])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)
    
    def __cleaning(self):
        word_net = WordNetLemmatizer()
        posts = Posts()
        clean_posts = Data(self.__tempFile, overwrite=True)
        for post in posts:
            # Remove HTML tags
            soup = BeautifulSoup(post, 'lxml')
            for tag in soup.find_all('code'):
                tag.decompose()
            post = soup.get_text()
            # Tokenize, remove stopwords and lemmatize
            post = [ word_net.lemmatize(token, pos=self.__getPOS(token)) for token in simple_preprocess(post, deacc=True) if token not in STOPWORDS ]
            post = ' '.join(post)
            clean_posts.append(post)

    def __enrichment(self):
        # Train Phrases model
        clean_posts = Data(self.__tempFile)
        bigram_model = Phrases(clean_posts, min_count=1)

        # Make bi-grams
        pre_processed_posts = PreProcessedPosts(overwrite=True)
        for post in clean_posts:
            # Concatenate bi-grams
            post = post.split()
            bigrams = [ bigram for bigram in bigram_model[post] if '_' in bigram ]
            post += bigrams
            post = ' '.join(post)
            pre_processed_posts.append(post)
        # Remove temporary file
        remove(self.__tempFile)

    def _process(self):
        self.__cleaning()
        self.__enrichment()