import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder

class FarmTextEncoder(BaseTorchEncoder):
    """
    Encode an array of string in size `B` into an ndarray in size `B x D`

    The ndarray potentially is BatchSize x (Channel x Height x Width)
    into an ndarray in size `B x D`. Where `B` is the batch size and
    `D` is the Dimension.

    FARM-based text encoder: (Framework for Adapting Representation Models)
    https://github.com/deepset-ai/FARM

    :param model_name_or_path: Directory or public name of the model to load.
    :param num_processes: Number of processes for `multiprocessing.Pool`.
        Set to value of 0 to disable multiprocessing. Set to None to let
        Inferencer use all CPU cores. If you want to debug the Language Model,
        you might need to disable multiprocessing
    :param extraction_strategy: Strategy to extract vectors. Choices:
        'cls_token' (sentence vector), 'reduce_mean'(sentence vector),
        reduce_max (sentence vector), 'per_token' (individual token vectors)
    :param extraction_layer: number of layer from which the embeddings shall
        be extracted. Default: -1 (very last layer).
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, pretrained_model_name_or_path: str = 'deepset/bert-base-cased-squad2',
                 num_processes: int = 0, extraction_strategy: str = 'cls_token',
                 extraction_layer: int = -1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.num_processes = num_processes
        self.extraction_strategy = extraction_strategy
        self.extraction_layer = extraction_layer

    def post_init(self):
        """Load FARM-based text model"""
        from farm.infer import Inferencer
        self.model = Inferencer.load(model_name_or_path=self.pretrained_model_name_or_path, task_type='embeddings',
                                     num_processes=self.num_processes)

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode an array of string in size `B` into an ndarray in size `B x D`.

        The ndarray potentially is BatchSize x (Channel x Height x Width)
            into an ndarray in size `B x D`. Where `D` is the Dimension.

        :param data: a 1d array of string type in size `B`
        :return: an ndarray in size `B x D`.
        """
        basic_texts = [{'text': s} for s in data]
        embeds = np.stack([k['vec'] for k in self.model.extract_vectors(dicts=basic_texts)])
        return embeds