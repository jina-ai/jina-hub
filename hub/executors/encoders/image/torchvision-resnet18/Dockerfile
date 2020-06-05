FROM pytorch/pytorch:latest

RUN pip install jina
RUN python -c "import torchvision.models as models; models.resnet18(pretrained=True, progress=True)"

ADD torchvision_resnet18.py manifest.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "torchvision_resnet18.py"]
