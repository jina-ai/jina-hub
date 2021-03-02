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
    crafted_doc = crafter.craft(signal_orig, 0)

    signal_mono = crafted_doc["blob"]
    assert signal_mono.shape[0] == signal_orig.shape[1]
