__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
from jina.executors.decorators import batching, as_ndarray
from jina.executors.encoders import BaseAudioEncoder


class MFCCTimbreEncoder(BaseAudioEncoder):
    """
    Extract a `n_mfcc`-dimensional feature vector for each MFCC frame.

    :class:`MFCCTimbreEncoder` is based on Mel-Frequency Cepstral
        Coefficients (MFCCs) which represent timbral features.
    :class:`MFCCTimbreEncoder` encodes an audio signal from a
        `Batch x Signal Length` ndarray into a
        `Batch x Concatenated Features` ndarray.

    :param input_sample_rate: input sampling rate in Hz
        (22050 by default)
    :param n_mfcc: the number of coefficients
        (20 by default)
    :param n_fft: length of the FFT window
        (2048 by default)
    :param hop_length: the number of samples between
        successive MFCC frames (512 by default)
    """

    def __init__(self, input_sample_rate: int = 22050, n_mfcc: int = 20, n_fft_length: int = 2048,
                 hop_length: int = 512, *args, **kwargs):
        """Set Constructor."""
        super().__init__(*args, **kwargs)
        self.input_sample_rate = input_sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft_length = n_fft_length
        self.hop_length = hop_length

    @batching
    @as_ndarray
    def encode(self, content: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Craft the audio signal of each Document into short MFCC frames.

        Extract MFCCs for each frame and concatenates Document frame
            MFCCs into a single Document embedding.

        :param content: a `Batch x Signal Length` ndarray,
            where `Signal Length` is a number of samples
        :return: a `Batch x Concatenated Features` ndarray,
            where `Concatinated Features` is a `n_mfcc`-dimensional
            feature vector times the number of the MFCC frames
        """
        from librosa.feature import mfcc
        embeds = []
        for chunk_data in content:
            mfccs = mfcc(y=chunk_data, sr=self.input_sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft_length,
                         hop_length=self.hop_length)
            embeds.append(mfccs.flatten())
        return embeds
