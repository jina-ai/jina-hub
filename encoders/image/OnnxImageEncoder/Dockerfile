FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y git curl && \
    pip install -r requirements.txt

# RUN pip install -r requirements.txt
COPY . /

# for downloading pre-trained model and testing the image
RUN bash download.sh && pip install pytest && pytest -v -s && rm -rf pretrained

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]