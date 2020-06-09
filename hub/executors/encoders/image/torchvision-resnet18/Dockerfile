FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel


RUN pip install jina && \
    pip install torch && \
    pip install torchvision
RUN python -c "import torchvision.models as models; model_name = models.resnet18(pretrained=True)"


ENTRYPOINT ["jina", "pod", "--yaml-path", "torchvision_resnet18.py"]
