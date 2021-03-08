__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np

from .. import AudioMonophoner


def test_audiomonophoner():
    """
    Tests crafter signal with monophoner and original signal have expected shapes
    """
    signal_orig = np.random.randn(2, 31337)

    crafter = AudioMonophoner()
    crafted_docs = crafter.craft(np.stack([signal_orig, signal_orig]))

    assert len(crafted_docs) == 2

    signal_mono = crafted_docs[0]['blob']
    assert signal_mono.shape[0] == signal_orig.shape[1]

    signal_mono = crafted_docs[1]['blob']
    assert signal_mono.shape[0] == signal_orig.shape[1]
