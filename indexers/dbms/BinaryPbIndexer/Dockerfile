FROM jinaai/jina:2.0.0

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y gcc build-essential

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# for testing the image
RUN pip install pytest && pytest tests/test_indexer.py

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
