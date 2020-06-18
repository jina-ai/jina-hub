FROM jinaai/jina:devel

ADD *.py arrayreader.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "arrayreader.yml"]

