FROM jinaai/jina:1.2.1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN python -c "import tensorflow as tf; net = tf.keras.applications.MobileNetV2(); net.load_weights('/root/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'); net.save('models/mobilenet_v2.h5')"

RUN pip install pytest && pytest -v -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]