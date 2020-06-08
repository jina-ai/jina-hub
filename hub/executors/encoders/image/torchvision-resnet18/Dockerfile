FROM pytorch/pytorch:latest

RUN pip install jina
RUN python -c "import torchvision.models as models; model_name = models.resnet18(pretrained=True)"


ENTRYPOINT ["jina", "pod", "--yaml-path", "torchvision_resnet18.py"]
