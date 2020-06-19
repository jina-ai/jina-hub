FROM onnx/onnx-base:latest

RUN pip install onnxruntime
RUN pip install jina

ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "onnx.yml"]