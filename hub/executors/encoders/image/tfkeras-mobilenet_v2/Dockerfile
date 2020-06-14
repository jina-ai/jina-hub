FROM tensorflow/tensorflow:latest

RUN python -c "import tensorflow as tf; img_shape=96; pool_strategy='avg'; weights_type='imagenet'; tf.keras.applications.MobileNetV2(input_shape=(img_shape, img_shape, 3), include_top=False, pooling=pool_strategy, weights=weights_type)"
RUN pip install jina

ENTRYPOINT ["jina", "pod", "--yaml-path", "KerasImageEncoder"]