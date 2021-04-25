FROM jinaai/jina:1.1.6 as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
FROM base
RUN pip install pytest && pytest -s -vv

FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]