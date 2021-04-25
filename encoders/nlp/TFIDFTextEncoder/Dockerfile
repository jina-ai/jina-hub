FROM jinaai/jina:1.1.6 as base

# install git
RUN apt-get -y update && apt-get install -y git

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -U scikit-learn

# for testing the image
FROM base
RUN pip install pytest && pytest -s -vv

FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]