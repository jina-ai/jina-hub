from jina.executors.encoders.image.torchvision import ImageTorchEncoder

class TorchVisionResNet18(ImageTorchEncoder):

    def __init__(self, *args, **kwargs):
        from torchvision.models import resnet18
        super().__init__(*args, **kwargs)
        self.model_name = resnet18(pretrained=True, progress=True)
        self.pool_strategy = 'max'


