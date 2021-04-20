FROM jinaai/jina:1.1.8 as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# for testing the image
FROM base
RUN pip install pytest && pytest

FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]