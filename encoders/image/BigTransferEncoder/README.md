# BigTransferEncoder

`BigTransferEncoder` is Big Transfer (BiT) presented by Google (https://github.com/google-research/big_transfer), this class use pretrained BiT to encode data from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally, :class:`BigTransferEncoder` wraps the models from https://storage.googleapis.com/bit_models/.