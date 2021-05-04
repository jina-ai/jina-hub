FROM jinaai/jina:1.2.1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y git curl git-lfs

RUN pip install -r requirements.txt

COPY . /

RUN git clone https://github.com/onnx/models.git && \
    cd models && \
    # need to have this checkout, otherwise model in test not loaded
    git checkout cbda9ebd037241c6c6a0826971741d5532af8fa4 && \
    git lfs install && \
    git lfs pull --include="vision/classification/mobilenet/model/mobilenetv2-7.onnx" --exclude="" && \
    cd ..

# for downloading pre-trained model and testing the image
RUN pip install pytest && pytest -v -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]