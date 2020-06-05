FROM pytorch/pytorch:latest

RUN pip install jina
RUN python -c "import torchvision.models as models; models.densenet161(pretrained=True, progress=True)"

ADD torchvision_densenet161.py manifest.yml ./

ENTRYPOINT ["jina", "pod", "--yaml-path", "torchvision_densenet161.py"]
