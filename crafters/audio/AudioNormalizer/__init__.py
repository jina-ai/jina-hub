import numpy as np

from jina.executors.crafters import BaseCrafter
from jina.executors.decorators import single


class AudioNormalizer(BaseCrafter):
    """:class:`AudioNormalizer` normalizes the audio signal on doc-level."""

    @single
    def craft(self, blob: 'np.ndarray', *args, **kwargs):
        """
        Normalize signal from the audio signal.

        Reads the `ndarray` of the audio signal,
        normalizes the signal and saves the `ndarray` of the
        normalized signal in the `blob` of the Document.

        :param blob: the ndarray of the audio signal
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: a Document dict with the normalized audio signal
        """
        import librosa
        signal_norm = librosa.util.normalize(blob)

        return dict(offset=0, blob=signal_norm)
