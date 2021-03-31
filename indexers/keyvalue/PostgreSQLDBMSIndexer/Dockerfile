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

RUN adduser --disabled-password --gecos '' postgres
RUN adduser postgres sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER postgres

FROM base
RUN initdb /usr/local/var/default_db &&\
    pg_ctl -D /usr/local/var/default_db start &&\
    psql --command "CREATE USER default_name WITH SUPERUSER PASSWORD 'default_pwd';" &&\
    createdb -O default_name default_db

FROM base
EXPOSE 5432

FROM base
RUN pip install pytest && pytest 


# jina pod
ENTRYPOINT [ "jina", "pod", "--uses", "config.yml" ]
