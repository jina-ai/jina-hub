# ImageOnnxEncoder

`ImageOnnxEncoder` encodes document content from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally :class:`ImageOnnxEncoder` wraps the models from `onnxruntime`.