FROM jinaai/jina:1.2.1 AS base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt --no-cache-dir

# for testing the image
FROM base

RUN pip install pytest && pytest

FROM base

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]