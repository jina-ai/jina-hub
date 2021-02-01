FROM mongo AS base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install pip
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv python3-pip

# install the third-party requirements
RUN python3.8 -m pip install -r requirements.txt

FROM base
# for testing the image
RUN mongod --fork -logpath /var/log/mongod.log && python3.8 -m pip install pytest && pytest -s

FROM base
# starts mongod as a deamon & jina pod
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
