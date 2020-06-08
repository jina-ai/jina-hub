FROM pytorch/pytorch:latest

RUN pip install jina

RUN python -c "import paddlehub as hub; x='xception71_imagenet', hub.Module(name=x)"

ENTRYPOINT ["jina", "pod", "--yaml-path", "ImagePaddlehubEncoder"]

