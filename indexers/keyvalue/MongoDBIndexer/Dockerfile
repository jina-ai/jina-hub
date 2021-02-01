FROM mongo AS base

COPY . /workspace

# install pip
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv python3-pip

# install the third-party requirements
RUN python3.8 -m pip install -r requirements.txt

# setup the workspace
WORKDIR /workspace

FROM base
# for testing the image
RUN mongod --fork -logpath /var/log/mongod.log && python3.8 -m pip install pytest && pytest

FROM base

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
