FROM mongo:4.4.3-bionic

# install and upgrade pip
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip
RUN python3.8 -m pip install --upgrade pip

# setup the workspace
COPY requirements.txt /requirements.txt

# install the third-party requirements
RUN python3.8 -m pip install -r requirements.txt

COPY . /workspace
WORKDIR /workspace

# for testing the image
RUN mongod --fork -logpath /var/log/mongod.log && python3.8 -m pip install pytest && pytest -s

# jina pod
ENTRYPOINT [ "jina", "pod", "--uses", "config.yml" ]
