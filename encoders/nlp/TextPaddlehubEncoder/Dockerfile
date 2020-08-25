FROM jinaai/jina:devel

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN apt-get update && \
    apt-get -y install libgomp1 libgl1-mesa-glx libglib2.0-0

# install the third-party requirements
RUN pip install -r requirements.txt && pip uninstall -y pathlib

# for testing the image
RUN pip install -y pytest pytest-mock mock && JINA_TEST_PRETRAINED='true' pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]