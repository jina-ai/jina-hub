FROM jinaai/jina AS base

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip && pip install -r requirements.txt

# for testing the image
FROM base
RUN pip install pytest && pytest

FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]