FROM jinaai/jina:devel

RUN apt-get update && apt-get install --no-install-recommends -y g++
RUN pip install annoy

ADD *.py *.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "annoy.yml"]