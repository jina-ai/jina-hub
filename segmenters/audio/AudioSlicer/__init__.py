from typing import Dict, List

import numpy as np
from jina.executors.segmenters import BaseSegmenter


class AudioSlicer(BaseSegmenter):
    """
    :class:`AudioSlicer` segments the audio signal on the doc-level into frames on the chunk-level.
    """

    def __init__(self, frame_length: int = 2048, hop_length: int = 512, *args, **kwargs):
        """
        :param frame_size: the number of samples in each frame
        """
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length

    def _segment(self, signal):
        import librosa
        if signal.ndim == 1:  # mono
            frames = librosa.util.frame(signal, frame_length=self.frame_length, hop_length=self.hop_length, axis=0)
        elif signal.ndim == 2:  # stereo
            left_frames = librosa.util.frame(
                signal[0,], frame_length=self.frame_length, hop_length=self.hop_length, axis=0)
            right_frames = librosa.util.frame(
                signal[1,], frame_length=self.frame_length, hop_length=self.hop_length, axis=0)
            frames = np.concatenate((left_frames, right_frames), axis=0)
        else:
            raise ValueError(f'audio signal must be 1D or 2D array: {signal}')
        return frames

    def segment(self, blob: 'np.ndarray', *args, **kwargs) -> List[Dict]:
        """
        Slices the input audio signal array into frames and saves the `ndarray` of each frame in the `blob` of each
        Chunk.

        :param blob: the ndarray of the audio signal
        :return: a list of Chunk dicts with audio frames
        """
        frames = self._segment(blob)

        return [dict(offset=idx, weight=1.0, blob=frame, length=frames.shape[0])
                for idx, frame in enumerate(frames)]

