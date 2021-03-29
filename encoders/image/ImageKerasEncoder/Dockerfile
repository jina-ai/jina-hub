FROM jinaai/jina:1.0.16

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# RUN pip install -r requirements.txt
RUN pip install -r requirements.txt

# for testing the image
RUN pip install pytest && pytest

ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]