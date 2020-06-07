FROM pytorch/pytorch:latest

RUN pip install jina
RUN python -c "import paddlehub.models as models; models.xception71_imagenet(trainable=False)"

ENTRYPOINT ["jina", "pod", "--yaml-path", "ImagePaddlehubEncoder"]

