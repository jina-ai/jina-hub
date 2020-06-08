from jina.executors.encoders.image.torchvision import ImageTorchEncoder


class TorchVisionResNet18(ImageTorchEncoder):

    def __init__(self, *args, **kwargs):
        from torchvision.models import resnet18

        self.model_name = resnet18(pretrained=True, progress=True)
        self.pool_strategy = 'max'

        super().__init__(self.model_name, self.pool_strategy, *args, **kwargs)
