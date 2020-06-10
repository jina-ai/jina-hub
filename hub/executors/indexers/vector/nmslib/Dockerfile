FROM jinaai/jina:devel

RUN pip install nmslib

RUN pip install jina

ENV TMP_WORKSPACE /tmp/jina/nmslib/
ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "nmslib.yml"]