__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import numpy as np

from jina.executors.encoders import BaseEncoder
from jina.excepts import PretrainedModelFileDoesNotExist

cur_dir = os.path.dirname(os.path.abspath(__file__))


class TFIDFTextEncoder(BaseEncoder):
    """Encode ``Document`` content from a `np.ndarray` (of strings) of length `BatchSize` into
    a `csr_matrix` of shape `Batchsize x EmbeddingDimension`.

    :param path_vectorizer: path containing the fitted tfidf encoder object
    :param args: not used
    :param kwargs: not used
    """

    def __init__(
        self,
        path_vectorizer: str = os.path.join(cur_dir, 'model/tfidf_vectorizer.pickle'),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.path_vectorizer = path_vectorizer

    def post_init(self):
        import os
        import pickle

        super().post_init()
        if os.path.exists(self.path_vectorizer):
            self.tfidf_vectorizer = pickle.load(open(self.path_vectorizer, 'rb'))
        else:
            raise PretrainedModelFileDoesNotExist(
                f'{self.path_vectorizer} not found, cannot find a fitted tfidf_vectorizer'
            )

    def encode(self, content: np.ndarray, *args, **kwargs) -> 'scipy.sparse.csr_matrix':
        """Encode the ``Document`` content creating a tf-idf feature vector of the input.

        :param content: numpy array of strings containing the text data to be encoded
        :param args: not used
        :param kwargs: not used
        """
        return self.tfidf_vectorizer.transform(content)
