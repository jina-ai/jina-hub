from jina.executors.encoders.image.torchvision import ImageTorchEncoder

class TorchVisionResNet18(ImageTorchEncoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = 'densenet161'
        self.pool_strategy = 'mean'