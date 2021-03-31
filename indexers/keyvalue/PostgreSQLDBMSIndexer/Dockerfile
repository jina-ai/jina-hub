FROM postgres:13.2 as base

# install and upgrade pip
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3-pip
RUN python3.7 -m pip install --upgrade pip

# setup the workspace
COPY requirements.txt /requirements.txt

# install Jina and third-party requirements
RUN python3.7 -m pip install -r requirements.txt

COPY . /workspace
WORKDIR /workspace

RUN sudo initdb /usr/local/var/postgres &&\
    sudo pg_ctl -D /usr/local/var/postgres start

FROM base
RUN pip install pytest && pytest 

FROM base
EXPOSE 5432

# jina pod
ENTRYPOINT [ "jina", "pod", "--uses", "config.yml" ]
