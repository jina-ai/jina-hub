from typing import Dict, List

import numpy as np
from jina.executors.decorators import single
from jina.executors.segmenters import BaseSegmenter


class AudioSlicer(BaseSegmenter):
    """
    :class:`AudioSlicer` segments the audio signal on the doc-level into frames on the chunk-level.

    :param frame_length: the number of samples in each frame
    :param hop_length: the number of steps to advance between frames
    """

    def __init__(self, frame_length: int = 2048, hop_length: int = 512, *args, **kwargs):
        """Set constructor"""
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length

    def _segment(self, signal):
        import librosa
        if signal.ndim == 1:  # mono
            frames = librosa.util.frame(signal, frame_length=self.frame_length, hop_length=self.hop_length, axis=0)
            return frames,

        elif signal.ndim == 2:  # stereo
            left_frames = librosa.util.frame(
                signal[0,], frame_length=self.frame_length, hop_length=self.hop_length, axis=0)
            right_frames = librosa.util.frame(
                signal[1,], frame_length=self.frame_length, hop_length=self.hop_length, axis=0)
            return left_frames, right_frames

        else:
            raise ValueError(f'audio signal must be 1D or 2D array: {signal}')

    @single
    def segment(self, blob: 'np.ndarray', *args, **kwargs) -> List[Dict]:
        """
        Slices the input audio signal array into frames
        and saves the `ndarray` of each frame in the `blob` of each sub-document.

        :param blob: the ndarray of the audio signal
        :return: a list of Chunk dicts with audio frames
        """
        channel_frames = self._segment(blob)

        chunks = []

        channel_tags = ('mono',) if len(channel_frames) == 1 else ('left', 'right')

        for frames, tag in zip(channel_frames, channel_tags):
            start = 0
            for idx, frame in enumerate(frames):
                chunks.append(dict(offset=idx, weight=1.0, blob=frame, location=[start, start + len(frame)],
                                   tags={'channel': tag}))
                start += self.hop_length

        return chunks
