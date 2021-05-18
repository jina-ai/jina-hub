FROM jinaai/jina:1.2.0

# setup the workspace
COPY . /workspace
WORKDIR /workspace

RUN apt-get update -y && apt-get install -y git

RUN pip install pip install git+https://github.com/jina-ai/jina.git@0f4afa3d538fd1a52bf28ada0647e851e339b970

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]