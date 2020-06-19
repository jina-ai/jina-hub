FROM jinaai/jina:devel

RUN pip install nmslib

ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "nmslib.yml"]