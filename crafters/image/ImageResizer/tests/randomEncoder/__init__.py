from jina.executors.encoders import BaseEncoder

class randomEncoder(BaseEncoder):
    """
    :class:`randomEncoder` random things.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # your customized __init__ below
        raise NotImplementedError

    

    def encode(self, data, *args, **kwargs):
        raise NotImplementedError

    
