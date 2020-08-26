# OnnxImageEncoder

`OnnxImageEncoder` encodes data from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally :class:`OnnxImageEncoder` wraps the models from `onnxruntime`.