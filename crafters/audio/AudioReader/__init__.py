from typing import Dict

from jina.executors.crafters import BaseCrafter
from jina.executors.decorators import single


class AudioReader(BaseCrafter):
    """
    Read and resample the audio signal on doc-level.

    :class:`AudioReader` loads an audio file as `ndarray` and resamples the audio signal to the target sampling rate
    (default 22050Hz).

    :param target_sample_rate: target sampling rate (scalar number > 0)
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(self, target_sample_rate: int = 22050, *args, **kwargs):
        """Set constructor."""
        super().__init__(*args, **kwargs)
        self.sample_rate = target_sample_rate

    @single
    def craft(self, uri: str, *args, **kwargs) -> Dict:
        """
        Decode given audio file and resample signal.

        Save the `ndarray` of the signal in the `blob`
        of the Document.

        Supported sound formats: WAV, MP3, OGG, AU,
        FLAC, RAW, AIFF, AIFF-C, PAF, SVX, NIST, VOC,
        IRCAM, W64, MAT4, MAT5, PVF, XI, HTK, SDS,
        AVR, WAVEX, SD2, CAF, WVE, MPC2K, RF64.

        :param uri: the audio file path.
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: a Document dict with the decoded audio signal
        """
        import librosa
        signal, orig_sr = librosa.load(uri, sr=self.sample_rate, mono=False)

        return dict(blob=signal)
