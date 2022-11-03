from gensim.models import Word2Vec, FastText
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer


class FeatureExtraction:
    def __init__(self, data):
        self.data = data

    def train_word2vec(self, size=100, window=5, min_count=5, workers=4, epochs=100):
        """Train Word2Vec model on the given data.

        Args:
            size (int, optional): Dimensionality of the word vectors.. Defaults to 100.
            window (int, optional): Maximum distance between the current and predicted word within a sentence.. Defaults to 5.
            min_count (int, optional): Ignores all words with total frequency lower than this. Defaults to 5.
            workers (int, optional): Use these many worker threads to train the model (=faster training with multicore machines). Defaults to 4.
            epochs (int, optional): Number of iterations (epochs) over the corpus. Defaults to 100.

        Returns:
            model.wv: Trained word2vec model vectors.
        """        
        data = list(map(lambda x: x.split(), self.data))
        model = Word2Vec(vector_size=size, window=window,
                         min_count=min_count, workers=workers, epochs=epochs)
        model.build_vocab(data, progress_per=10000)
        model.train(data, total_examples=model.corpus_count, epochs=epochs, report_delay=1)
        return model.wv

    def train_fasttext(self, size=100, window=5, min_count=5, workers=4, epochs=5):
        """Train FastText model on the given data.

        Args:
            size (int, optional): Dimensionality of the word vectors.. Defaults to 100.
            window (int, optional): Maximum distance between the current and predicted word within a sentence.. Defaults to 5.
            min_count (int, optional): Ignores all words with total frequency lower than this. Defaults to 5.
            workers (int, optional): Use these many worker threads to train the model (=faster training with multicore machines). Defaults to 4.
            epochs (int, optional): Number of iterations (epochs) over the corpus. Defaults to 100.

        Returns:
            model.wv: Trained FastText model vectors.
        """ 
        data = list(map(lambda x: x.split(), self.data))
        model = FastText(data, vector_size=size, window=window,
                         min_count=min_count, workers=workers, epochs=epochs)
        model.build_vocab(data, progress_per=10000)
        model.train(data, total_examples=model.corpus_count, epochs=epochs, report_delay=1)
        return model.wv

    def use_pretrained_embs(self, embeddings_path):
        """Generate glove embeddings for the given model.

        Args:
            embeddings_path (str): Path to the pretrained embeddings.

        Returns:
            embed_dict: Dictionary of pretrained word embeddings.
        """
        embed_dict = {}
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], 'float32')
                embed_dict[word] = vector
        print('Found %s word vectors.' %len(embed_dict))
        return embed_dict

    # Acquire embeddings from a HF Transformer model
    def use_transformer_embs(self, transformer_model):
        """Generate transformer embeddings for the given model.

        Args:
            transformer_model (SentenceTransformer): Pretrained  HuggingFace transformer model.

        Returns:
            sentence_embeddings: List of sentence embeddings.
        """
        model = SentenceTransformer(transformer_model)
        sentence_embeddings = model.encode(self.data)
        return sentence_embeddings


    def get_max_len(self):
        """Get the maximum length of the sentences.

        Returns:
            max_len: Maximum length of the sentences.
        """
        max_len = 0
        for sentence in self.data:
            if len(sentence) > max_len:
                max_len = len(sentence)
        return max_len

    def get_word_index(self):
        """Get the word index.

        Returns:
            word_index: Dictionary of word index.
        """
        word_index = {}
        for sentence in self.data:
            for word in sentence.split():
                if word not in word_index:
                    word_index[word] = len(word_index) + 1
        return word_index

    def tfidf(self, max_features=1000, ngram_range=(1, 1)):
        """Train TF-IDF model on the given data.

        Args:
            max_features (int, optional): Maximum number of features to be extracted. Defaults to 1000.
            ngram_range (tuple, optional): Range of n-grams to be considered. Defaults to (1, 1).

        Returns:
            tfidf: Trained tfidf vectorizer object.
        """        """"""
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        tfidf.fit(self.data)
        return tfidf