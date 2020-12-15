FROM jinaai/jina:latest

# setup the workspace
COPY . /workspace

WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

RUN apt-get update && apt-get -y install redis-server

# for testing the image
RUN redis-server --bind 0.0.0.0 --port 63079:63079 --daemonize yes && pip install pytest && pip install pytest-mock && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]