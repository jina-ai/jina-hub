FROM jinaai/jina:1.1.6 as base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements

RUN apt-get update && apt-get install -y git && pip install git+https://github.com/facebookresearch/pysparnn.git && pip install -r requirements.txt

# for testing the image
FROM base
RUN pip install pytest && pytest -v -s

FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]