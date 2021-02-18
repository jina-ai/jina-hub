__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders.frameworks import BaseTorchEncoder


class VSEImageEncoder(BaseTorchEncoder):
    """
    :class:`VSEImageEncoder` encodes data from a ndarray, potentially BatchSize x (Channel x Height x Width) into a
        ndarray of BatchSize * d.

    The :clas:`VSEImageEncoder` was originally proposed in `VSE++: Improving Visual-Semantic Embeddings with Hard Negatives <https://github.com/fartashf/vsepp>`_.

    :param path: The VSE model path.
    :param pool_strategy: Pooling strategy for the encoder, default `mean`.
    :param channel_axis: The axis of the channel, default -1, will move the axis of input data from -1 to 1.
    :param args: additional positional arguments.
    :param kwargs: additional positional arguments.
    """

    def __init__(self,
                 path: str = 'runs/f30k_vse++_vggfull/model_best.pth.tar',
                 pool_strategy: str = 'mean',
                 channel_axis: int = 1,
                 *args,
                 **kwargs):
        """Class constructor."""
        super().__init__(*args, **kwargs)
        self.path = path
        self.pool_strategy = pool_strategy
        self.channel_axis = channel_axis
        self._default_channel_axis = 1

    def post_init(self):
        """Load VSE++ model."""
        import torch
        from .model import VSE

        if self.pool_strategy is not None:
            self.pool_fn = getattr(np, self.pool_strategy)

        checkpoint = torch.load(self.path,
                                map_location=torch.device('cpu' if not self.on_gpu else 'cuda'))
        opt = checkpoint['opt']

        model = VSE(opt)
        model.load_state_dict(checkpoint['model'])
        model.img_enc.eval()
        self.model = model.img_enc
        self.to_device(self.model)
        del model.txt_enc

    def _get_features(self, data):
        from torch.autograd import Variable
        # It needs Resize and Normalization before reaching this Point in another Pod
        # Check how this works, it may not be necessary to squeeze
        images = Variable(data, requires_grad=False)
        img_emb = self.model(images)
        return img_emb

    def _get_pooling(self, feature_map: 'np.ndarray') -> 'np.ndarray':
        if feature_map.ndim == 2 or self.pool_strategy is None:
            return feature_map
        return self.pool_fn(feature_map, axis=(2, 3))

    @batching
    @as_ndarray
    def encode(self, data: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode input data into `np.ndarray`.

        :param data: Image to be encoded, expected a `np.ndarray` of BatchSize x (Channel x Height x Width).
        :param args: additional positional arguments.
        :param kwargs: additional positional arguments.
        :return: Encoded result as `np.ndarray`.
        """
        if self.channel_axis != self._default_channel_axis:
            data = np.moveaxis(data, self.channel_axis, self._default_channel_axis)
        import torch
        _input = torch.from_numpy(data.astype('float32'))
        if self.on_gpu:
            _input = _input.cuda()
        _feature = self._get_features(_input).detach()
        if self.on_gpu:
            _feature = _feature.cpu()
        _feature = _feature.numpy()
        return self._get_pooling(_feature)
