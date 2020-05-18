FROM onnx/onnx-base

RUN pip install jina
RUN python -c "import onnx"

ENTRYPOINT ["jina", "pod", "--yaml-path", "OnnxImageEncoder"]

