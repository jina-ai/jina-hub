from jina.executors.encoders.image.torchvision import ImageTorchEncoder

class TorchVisionResNet18(ImageTorchEncoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'resnet18'
        self.pool_strategy = 'max'

