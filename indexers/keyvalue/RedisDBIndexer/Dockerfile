FROM redis:latest

CMD redis-server --bind 127.0.0.1 --port 27017

FROM jinaai/jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest


ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]