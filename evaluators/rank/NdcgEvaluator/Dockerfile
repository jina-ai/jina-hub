FROM jinaai/jina:1.2.2

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# for testing the image
RUN pip install pytest pytest-repeat && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]