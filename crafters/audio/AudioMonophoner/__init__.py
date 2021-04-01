__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict

import numpy as np

from jina.executors.crafters import BaseCrafter
from jina.executors.decorators import single


class AudioMonophoner(BaseCrafter):
    """:class:`AudioMonophoner` makes the audio signal monophonic on doc-level."""

    @single
    def craft(self, blob: 'np.ndarray', *args, **kwargs) -> Dict:
        """
        Read the `ndarray` of the audio signal.

        Makes the audio signal monophonic and saves the `ndarray` of the
        monophonic signal in the `blob` of the Document.

        :param blob: the ndarray of the audio signal
        :return: a Document dict with the monophonic audio signal
        :param args:  Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: A dictionary with the monophonic signal

        """
        import librosa
        signal_mono = librosa.to_mono(blob)

        return dict(offset=0, blob=signal_mono)

