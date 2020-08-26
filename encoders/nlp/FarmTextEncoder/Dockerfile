FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

ENV JINA_TEST_PRETRAINED='true'

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest pytest-mock mock && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
