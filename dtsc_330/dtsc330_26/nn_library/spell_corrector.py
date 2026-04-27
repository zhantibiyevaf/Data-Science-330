from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class SpellCorrector:
    """A lightweight character n-gram spelling corrector."""

    def __init__(self, ngram_range=(1, 3)):
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range)
        self.neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
        self.correct_words = None

    def fit(self, misspelled_words, correct_words):
        self.correct_words = list(correct_words)
        vectors = self.vectorizer.fit_transform(misspelled_words)
        self.neighbors.fit(vectors)
        return self

    def predict(self, words):
        if self.correct_words is None:
            raise ValueError("SpellCorrector must be fitted before calling predict().")

        vectors = self.vectorizer.transform(words)
        neighbor_indexes = self.neighbors.kneighbors(vectors, return_distance=False)
        return [self.correct_words[indexes[0]] for indexes in neighbor_indexes]
