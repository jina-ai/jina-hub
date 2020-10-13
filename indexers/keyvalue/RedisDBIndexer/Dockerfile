FROM jinaai/jina:latest

# setup the workspace
COPY . /workspace

WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt && apt-get update

RUN apt-get -y install redis-server

# for testing the image
RUN redis-server --bind 0.0.0.0 --port 63079:63079 --daemonize yes && pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]