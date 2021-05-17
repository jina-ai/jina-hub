FROM jinaai/jina:1.2.1

# install git
RUN apt-get -y update && apt-get install -y git

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install git+https://github.com/openai/CLIP.git

# for testing the image
RUN pip install pytest && pytest -s -vv

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]