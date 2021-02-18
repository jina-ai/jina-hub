FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install dependency and download pre-trained model
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl \
                                               gcc \
                                               g++ \
                                               libsndfile1 \
                                               python3-dev && \
    curl https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt --output /tmp/wav2vec_large.pt

# install the third-party requirements
RUN pip install -r requirements.txt
RUN pip uninstall dataclasses

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
