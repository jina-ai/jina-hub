# KerasImageEncoder

`KerasImageEncoder` encodes data from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`. Internally, :class:`KerasImageEncoder` wraps the models from `tensorflow.keras.applications`. https://keras.io/applications/