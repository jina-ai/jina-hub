FROM pytorch/pytorch:latest

ADD jina .src

RUN pip install .src/ .
RUN python -c "import torchvision.models as models; models.mobilenet_v2(pretrained=True)"

ENTRYPOINT ["jina", "pod", "--yaml-path", "ImageTorchEncoder"]