FROM jinaai/jina:1.0.16

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt

RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]