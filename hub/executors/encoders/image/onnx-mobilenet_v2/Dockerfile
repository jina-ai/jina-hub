FROM onnx/onnx-base:latest

RUN pip install onnxruntime
RUN pip install jina

ENV TMP_WORKSPACE /tmp/jina/onnx_mobilenet/
ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "onnx.yml"]