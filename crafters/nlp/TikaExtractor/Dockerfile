FROM logicalspark/docker-tikaserver

WORKDIR /

# get python
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# use python 3.7 as default
RUN update-alternatives --set python /usr/bin/python3.7

# get pip
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# install jina
RUN pip uninstall numpy -y
RUN pip install jina

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]