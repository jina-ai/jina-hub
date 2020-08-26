from jina.executors.encoders import BaseEncoder

class ChromaPitchEncoder(BaseEncoder):
    """
    :class:`ChromaPitchEncoder` encodes an audio signal from a `Batch x Signal Length` ndarray into a `Batch x Concatenated Features` ndarray..
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # your customized __init__ below
        raise NotImplementedError

    

    def encode(self, data, *args, **kwargs):
        raise NotImplementedError

    
