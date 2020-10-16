FROM jinaai/jina

COPY . /workspace
WORKDIR /workspace

RUN pip3 install -r requirements.txt
RUN pip3 install pytest && pytest tests/ -v -s

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]