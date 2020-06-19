FROM jinaai/jina:devel

RUN pip install ternausnet
ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "terausnet.yml"]