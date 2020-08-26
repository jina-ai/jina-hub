FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r --upgrade requirements.txt

# for testing the image
RUN pip install pytest pytest-mock mock && JINA_TEST_PRETRAINED='true' pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]