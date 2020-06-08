FROM pytorch/pytorch:latest

RUN pip install jina
RUN pip install torchvision
RUN python -c "import torchvision.models as models; model_name = models.densenet161(pretrained=True)"


ENTRYPOINT ["jina", "pod", "--yaml-path", "torchvision_densenet161.py"]
