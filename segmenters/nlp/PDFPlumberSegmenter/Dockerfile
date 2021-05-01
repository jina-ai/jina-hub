FROM jinaai/jina:1.1.5 as BASE

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
FROM BASE
RUN pip install pytest && pytest

FROM BASE
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]