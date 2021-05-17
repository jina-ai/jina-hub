FROM jinaai/jina:1.2.1 as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN apt-get update && \
    apt-get -y install libgomp1 libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# install the third-party requirements
RUN pip install -r requirements.txt && pip uninstall -y pathlib

# for testing the image
FROM base
RUN pip install pytest pytest-mock mock && pytest -v -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
