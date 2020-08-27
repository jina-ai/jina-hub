# VideoTorchEncoder

:class:`VideoTorchEncoder` encodes data from a ndarray, potentially B x T x (Channel x Height x Width) into an ndarray of `B x D`. Internally, :class:`VideoTorchEncoder` wraps the models from `torchvision.models`. https://pytorch.org/docs/stable/torchvision/models.html 