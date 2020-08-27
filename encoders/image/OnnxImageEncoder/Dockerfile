FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y git curl git-lfs

RUN pip install -r requirements.txt

COPY . /

RUN git-lfs clone https://github.com/onnx/models.git && \
    cd models && \
    git lfs install && \
    git lfs pull --include="vision/classification/mobilenet/model/mobilenetv2-7.onnx" --exclude="" && \
    cd ..

# for downloading pre-trained model and testing the image
RUN pip install pytest && pytest -v -s && rm -rf models

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]