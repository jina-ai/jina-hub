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

#RUN adduser postgres sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

#RUN mkdir -p /usr/local/var/
#RUN chown postgres /usr/local/var/

USER posgre

#RUN /etc/init.d/postgresql start &&\
#    /usr/pgsql-13.2/bin/postgresql132-setup initdb initdb /usr/local/var/default_db &&\
#    pg_ctl -D /usr/local/var/default_db start &&\
#    createdb default_db &&\
#    python3.7 -m pip install pytest && python3.7 -m pytest -s -v

EXPOSE 5432

#FROM base
#RUN pip install pytest && pytest

# jina pod
#ENTRYPOINT [ "jina", "pod", "--uses", "config.yml" ]
