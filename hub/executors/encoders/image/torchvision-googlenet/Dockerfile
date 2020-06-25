FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN pip install jina
RUN pip install scipy
RUN python -c "from torchvision.models import googlenet; model_name = googlenet(pretrained=True, progress=False, aux_logits=False, transform_input=True, **kwargs)"

ADD *.py *.yml ./

ENTRYPOINT["jina", "pod", "--yaml-path", "ImageTorchEncoder"]
