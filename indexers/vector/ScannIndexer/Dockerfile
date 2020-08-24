FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN apt-get update && apt-get install --no-install-recommends -y curl  && \
    curl https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp37-cp37m-linux_x86_64.whl --output scann-1.0.0-cp37-cp37m-linux_x86_64.whl

 RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]