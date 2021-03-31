FROM postgres:13.2 as base

# install and upgrade pip
RUN apt-get update && apt-get install -y python3.7 python3.7-dev python3-pip
RUN python3.7 -m pip install --upgrade pip

# setup the workspace
COPY requirements.txt /requirements.txt

# install Jina and third-party requirements
RUN python3.7 -m pip install -r requirements.txt
RUN python3.7 -m pip install pytest

COPY . /workspace
WORKDIR /workspace

ENV POSTGRES_PASSWORD=123456

RUN nohup bash -c "docker-entrypoint.sh postgres &" && sleep 3 && python3.7 -m pytest -s -v .

EXPOSE 5432

# TODO figure out how to start PSQL in the background for the entrypoint
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]
