import numpy as np
from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders import BaseAudioEncoder


class ChromaPitchEncoder(BaseAudioEncoder):
    """
    :class:`ChromaPitchEncoder` is based on chroma spectrograms (chromagrams) which represent melodic/harmonic features.
    :class:`ChromaPitchEncoder` encodes an audio signal from a `Batch x Signal Length` ndarray into a
    `Batch x Concatenated Features` ndarray.
    """

    def __init__(self, input_sample_rate: int = 22050, hop_length: int = 512, *args, **kwargs):
        """
        :class:`ChromaPitchEncoder` extracts from an audio signal a 12-dimensional feature vector, that refer to 12
        octaves, for each chroma frame.

        :param input_sample_rate: input sampling rate in Hz (22050 by default)
        :param hop_length: the number of samples between successive chroma frames (512 by default)
        """
        super().__init__(*args, **kwargs)
        self.input_sample_rate = input_sample_rate
        self.hop_length = hop_length

    @batching
    @as_ndarray
    def encode(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Segments the audio signal of each Chunk into short chroma frames, extracts chromagrams for each frame and
        concatenates Chunk frame chromagrams into a single Chunk embedding.

        :param data: a `Batch x Signal Length` ndarray, where `Signal Length` is a number of samples
        :return: a `Batch x Concatenated Features` ndarray, where `Concatinated Features` is a 12-dimensional feature
        vector times the number of the chroma frames
        """
        from librosa.feature import chroma_cqt
        embeds = []
        for chunk_data in data:
            chromagrams = chroma_cqt(y=chunk_data, sr=self.input_sample_rate, n_chroma=12, hop_length=self.hop_length)
            embeds.append(chromagrams.flatten())
        return embeds
