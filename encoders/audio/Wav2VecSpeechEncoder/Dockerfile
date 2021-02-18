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
# hotfix: upstream issues, https://github.com/google/flax/issues/269#issue-619773070
RUN pip uninstall -y dataclasses

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
