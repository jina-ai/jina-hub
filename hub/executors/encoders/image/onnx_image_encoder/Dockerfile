FROM nvidia/cuda:devel

RUN pip install jina
RUN pip install keras2onnx
RUN python -c "import keras; from keras.applications.resnet50 import ResNet50; model = ResNet50(include_top=True, weights='imagenet'); keras2onnx.convert_keras(model, model.name)"


ENTRYPOINT ["jina", "pod", "--yaml-path", "OnnxImageEncoder"]

