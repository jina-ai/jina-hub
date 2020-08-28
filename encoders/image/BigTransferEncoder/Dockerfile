FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y git curl && \
    pip install -r requirements.txt

COPY . /

# for downloading pre-trained model
RUN bash download.sh

# for testing the image
RUN pip install pytest && pytest -v

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]