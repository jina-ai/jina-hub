FROM jinaai/jina:1.2.1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y gcc build-essential
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
