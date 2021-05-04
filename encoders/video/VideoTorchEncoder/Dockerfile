FROM jinaai/jina:1.2.1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest pytest-mock mock && JINA_TEST_PRETRAINED='true' pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]