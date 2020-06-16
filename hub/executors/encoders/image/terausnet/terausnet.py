from jina.executors.encoders.frameworks import BaseCVTorchEncoder

class TernausNetEncoder(BaseCVTorchEncoder):
    """
    TernausNet by Vladimir Iglovikov and Alexey Shvets
    :class: TernausNetEncoder is a modification of the celebrated UNet architecture that is widely used for binary Image Segmentation
    """

    def __init__(self, *args, **kwargs):
        """
        :pretrained : vgg11
        """

        super().__init__(*args, **kwargs)

    def post_init(self):
        """
        Import terausnet model pretrained with vgg11
        """
        import ternausnet
        self.model = ternausnet.absolute_import