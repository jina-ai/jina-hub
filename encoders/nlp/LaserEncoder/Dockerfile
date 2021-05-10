FROM jinaai/jina:1.2.1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -m laserembeddings download-models

# for testing the image
RUN pip install pytest && JINA_TEST_PRETRAINED='true' pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]