FROM jinaai/jina:1.2.3 as base

RUN apt-get update && apt-get install libgomp1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
FROM base
RUN pip install pytest && pytest -s -v

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
