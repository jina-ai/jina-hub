FROM jinaai/jina:1.1.0

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]