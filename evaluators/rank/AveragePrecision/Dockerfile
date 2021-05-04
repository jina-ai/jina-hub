FROM jinaai/jina:1.2.1

COPY . /workspace
WORKDIR /workspace

RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]