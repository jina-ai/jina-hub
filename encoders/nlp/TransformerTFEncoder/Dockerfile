FROM jinaai/jina:1.3.0 as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
FROM base
RUN pip install pytest && pytest -v -s

FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]