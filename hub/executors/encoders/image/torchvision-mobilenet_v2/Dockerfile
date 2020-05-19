FROM pytorch/pytorch:latest

RUN pip install jina
RUN python -c "import torchvision.models as models; models.mobilenet_v2(pretrained=True)"

ENTRYPOINT ["jina", "pod", "--yaml-path", "ImageTorchEncoder"]

