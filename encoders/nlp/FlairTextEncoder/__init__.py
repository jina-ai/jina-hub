from typing import Union, Tuple, List

import numpy as np
from jina.executors.encoders.frameworks import BaseTorchEncoder


class FlairTextEncoder(BaseTorchEncoder):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    Internally, :class:`FlairTextEncoder` wraps the DocumentPoolEmbeddings from Flair.

    :param embeddings: the name of the embeddings. Supported models include
        - ``word:[ID]``: the classic word embedding model, the ``[ID]`` are listed at
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/CLASSIC_WORD_EMBEDDINGS.md
        - ``flair:[ID]``: the contextual embedding model, the ``[ID]`` are listed at
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
        - ``pooledflair:[ID]``: the pooled version of the contextual embedding model,
        the ``[ID]`` are listed at
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
        - ``byte-pair:[ID]``: the subword-level embedding model, the ``[ID]`` are listed at
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/BYTE_PAIR_EMBEDDINGS.md
        - ``Example``: ('word:glove', 'flair:news-forward', 'flair:news-backward')

    :param pooling_strategy: the strategy to merge the word embeddings into the chunk embedding.
    Supported strategies include ``mean``, ``min``, ``max``.
    """

    def __init__(self,
                 embeddings: Union[Tuple[str], List[str]] = ('word:glove', ),
                 pooling_strategy: str = 'mean',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = embeddings
        self.pooling_strategy = pooling_strategy
        self.max_length = -1  # reserved variable for future usages
        self._post_set_device = False

    def post_init(self):
        """
        Load model.

        Possible models are:
        - flair
        - pooledflair
        - word
        - byte-pair
        """
        import flair
        flair.device = self.device
        embeddings_list = []
        for e in self.embeddings:
            model_name, model_id = e.split(':', maxsplit=1)
            emb = None
            try:
                if model_name == 'flair':
                    from flair.embeddings import FlairEmbeddings
                    emb = FlairEmbeddings(model_id)
                elif model_name == 'pooledflair':
                    from flair.embeddings import PooledFlairEmbeddings
                    emb = PooledFlairEmbeddings(model_id)
                elif model_name == 'word':
                    from flair.embeddings import WordEmbeddings
                    emb = WordEmbeddings(model_id)
                elif model_name == 'byte-pair':
                    from flair.embeddings import BytePairEmbeddings
                    emb = BytePairEmbeddings(model_id)
            except ValueError:
                self.logger.error(f'embedding not found: {e}')
                continue
            if emb is not None:
                embeddings_list.append(emb)
        if embeddings_list:
            from flair.embeddings import DocumentPoolEmbeddings
            self.model = DocumentPoolEmbeddings(embeddings_list, pooling=self.pooling_strategy)
            self.logger.info(f'flair encoder initialized with embeddings: {self.embeddings}')
        else:
            self.logger.error('flair encoder initialization failed.')

    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode data from an array of string in size `B` into a ndarray in size `B x D`.

        :param data: a 1-dimension array of string type in size `B`
        :return: an ndarray in size `B x D`
        """
        from flair.data import Sentence
        c_batch = [Sentence(row) for row in data]
        self.model.embed(c_batch)
        result = [self.tensor2array(c_text.embedding) for c_text in c_batch]
        return np.vstack(result)

    def tensor2array(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy() if self.on_gpu else tensor.numpy()
