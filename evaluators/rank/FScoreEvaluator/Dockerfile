FROM jinaai/jina:1.1.0

COPY . /workspace
WORKDIR /workspace

RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]