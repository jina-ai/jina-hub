# CustomKerasImageEncoder

`CustomKerasImageEncoder` encodes data from a ndarray, potentially B x (Channel x Height x Width) into a ndarray of `B x D`.Internally, :class:`CustomKerasImageEncoder` wraps any custom tf.keras model not part of models from `tensorflow.keras.applications`. https://www.tensorflow.org/api_docs/python/tf/keras/applications