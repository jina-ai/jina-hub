import numpy as np
from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders import BaseAudioEncoder


class ChromaPitchEncoder(BaseAudioEncoder):
    """
    Segment  audio signal into short chroma frames.

    :class:`ChromaPitchEncoder` is based on chroma spectrograms
    (chromagrams) which represent melodic/harmonic features.
    :class:`ChromaPitchEncoder` encodes an audio signal from a
    `Batch x Signal Length` ndarray into a
    `Batch x Concatenated Features` ndarray.

    :param input_sample_rate: input sampling rate in Hz
        (22050 by default)
    :param hop_length: the number of samples between
        successive chroma frames (512 by default)
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, input_sample_rate: int = 22050, hop_length: int = 512, *args, **kwargs):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.input_sample_rate = input_sample_rate
        self.hop_length = hop_length

    @batching
    @as_ndarray
    def encode(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Craft audio signal of each Chunk into short chroma frames.

        Extract chromagrams for each frame and concatenates Chunk
        frame chromagrams into a single Chunk embedding.

        :param data: a `Batch x Signal Length` ndarray, where
            `Signal Length` is a number of samples
        :return: a `Batch x Concatenated Features` ndarray, where
            `Concatenated Features` is a 12-dimensional feature
            vector times the number of the chroma frames
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        """
        from librosa.feature import chroma_cqt
        embeds = []
        for chunk_data in data:
            chromagrams = chroma_cqt(y=chunk_data, sr=self.input_sample_rate, n_chroma=12, hop_length=self.hop_length)
            embeds.append(chromagrams.flatten())
        return embeds
