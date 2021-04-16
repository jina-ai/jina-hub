from typing import Optional
import numpy as np

from jina.executors.encoders.frameworks import BasePaddleEncoder
from jina.executors.decorators import batching, as_ndarray


class TextPaddlehubEncoder(BasePaddleEncoder):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)

    Internally, :class:`TextPaddlehubEncoder` wraps the Ernie module from paddlehub.
    https://github.com/PaddlePaddle/PaddleHub
    For models' details refer to
    https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SemanticModel

    :param model_name: the name of the model. Supported models include
        ``ernie``, ``ernie_tiny``, ``ernie_v2_eng_base``, ``ernie_v2_eng_large``,
        ``bert_chinese_L-12_H-768_A-12``, ``bert_multi_cased_L-12_H-768_A-12``,
        ``bert_multi_uncased_L-12_H-768_A-12``, ``bert_uncased_L-12_H-768_A-12``,
        ``bert_uncased_L-24_H-1024_A-16``, ``chinese-bert-wwm``,
        ``chinese-bert-wwm-ext``, ``chinese-electra-base``,
        ``chinese-electra-small``, ``chinese-roberta-wwm-ext``,
        ``chinese-roberta-wwm-ext-large``, ``rbt3``, ``rbtl3``
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, model_name: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name or 'ernie_tiny'

    def post_init(self):
        """Load PaddleHub model"""
        import paddlehub as hub
        self.model = hub.Module(name=self.model_name)

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode data from an array of string in size `B` into a ndarray in size `B x D`.

        :param data: a 1d array of string type in size `B`
        :return: an ndarray in size `B x D`
        """
        results = []
        _raw_results = self.model.get_embedding(
            np.atleast_2d(data).reshape(-1, 1).tolist(), use_gpu=self.on_gpu)
        for emb in _raw_results:
            _pooled_feature, _seq_feature = emb
            results.append(_pooled_feature)
        return np.array(results)
