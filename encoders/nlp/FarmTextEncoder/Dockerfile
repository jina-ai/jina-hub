FROM jinaai/jina:1.3.0

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN apt-get update && \
    apt-get install --no-install-recommends -y gcc\
                                               python-dev

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest pytest-mock mock && JINA_TEST_PRETRAINED='true' pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
