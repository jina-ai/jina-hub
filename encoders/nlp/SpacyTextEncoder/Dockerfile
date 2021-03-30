FROM jinaai/jina

# install git
RUN apt-get -y update && apt-get install -y git

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements, use en_core_web_sm for tok2vec encoder
RUN pip install -r requirements.txt https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz#egg=en_core_web_sm


# for testing the image
RUN pip install pytest && pytest -s -vv

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
