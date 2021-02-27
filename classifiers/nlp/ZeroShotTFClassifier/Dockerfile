FROM jinaai/jina as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install third-party requirements
RUN pip install -r requirements.txt

# for testing the image
FROM base as test
RUN pip install pytest && pytest

FROM base as final
ENTRYPOINT ['jina', 'pod', '--uses', 'config.yml']