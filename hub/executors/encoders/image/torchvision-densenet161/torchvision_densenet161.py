from jina.executors.encoders.image.torchvision import ImageTorchEncoder


class TorchVisionResNet18(ImageTorchEncoder):

    def __init__(self, *args, **kwargs):
        from torchvision.models import densenet161

        self.model_name = densenet161(pretrained=True)
        self.pool_strategy = 'mean'

        super().__init__(self.model_name, self.pool_strategy, *args, **kwargs)
