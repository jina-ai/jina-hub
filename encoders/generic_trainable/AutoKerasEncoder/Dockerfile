FROM jinaai/jina:1.2.1

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install -r requirements.txt


RUN pip install pytest && pytest -v -s


ENTRYPOINT ["jina", "pod", "--uses", "config_bert.yml"]
