FROM onnx/onnx-base

ADD jina .src

RUN pip install .src/.
RUN python -c "import onnx"

ENTRYPOINT ["jina", "pod", "--yaml-path", "OnnxImageEncoder"]

