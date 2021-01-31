FROM jinaai/jina as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install tensorflow==2.4.0 transformers==4.1.1 grpcio==1.34.0

FROM base
# for testing the image
RUN pip install pytest && pytest

FROM base

ENTRYPOINT ["jina", "pod", "--uses", "config.yml", "--timeout-ready", "180000"]