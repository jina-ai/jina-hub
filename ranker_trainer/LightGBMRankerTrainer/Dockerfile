FROM jinaai/jina:1.2.3 as base

RUN apt-get update

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
FROM base
RUN pip install pytest && pytest -v -s

FROM base
RUN pip install pytest &&
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
