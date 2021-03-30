FROM jinaai/jina:1.1.0

# setup the workspace
COPY . /workspace
WORKDIR /workspace


RUN apt-get update && \
    apt-get install --no-install-recommends -y libsndfile1

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]