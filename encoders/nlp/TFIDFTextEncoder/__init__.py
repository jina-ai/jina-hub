__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina.executors.encoders import BaseEncoder

class TFIDFTextEncoder(BaseEncoder):
    """Encode data from a `np.ndarray` (of strings) of length `BatchSize` into
    a `csr_matrix` of shape `Batchsize x EmbeddingDimension`. 

    :param path_vectorizer: path containing the fitted tfidf encoder object
    :param args: not used
    :param kwargs: not used
    """

    def __init__(self,
                 path_vectorizer= "./model/tfidf_vectorizer.pickle",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.path_vectorizer = path_vectorizer

    def post_init(self):
        import pickle
        self.tfidf_vectorizer = pickle.load(open(self.path_vectorizer, "rb"))

    def encode(self, data, *args, **kwargs) -> 'scipy.sparse.csr_matrix':
        """Encode the data creating a tf-idf feature vector of the input.

        :param data: text data to be encoded
        :param args: not used
        :param kwargs: not used
        """
        return self.tfidf_vectorizer.transform(data)