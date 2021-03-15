FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime as base

# install git
RUN apt-get -y update && apt-get install -y git

# setup the workspace
COPY . /workspace
WORKDIR /workspace


RUN pip install jina
# install the third-party requirements
RUN pip install git+https://github.com/openai/CLIP.git


# for testing the image
FROM base
RUN pip install pytest && pytest


FROM base
ENTRYPOINT ["jina", "pod", "--uses", "config.yml"]