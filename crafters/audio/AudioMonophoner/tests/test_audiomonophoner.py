import numpy as np

from .. import AudioMonophoner


def test_audiomonophoner():
    """here is my test code

    https://docs.pytest.org/en/stable/getting-started.html#create-your-first-test
    """
    signal_orig = np.random.randn(2, 31337)

    crafter = AudioMonophoner()
    crafted_doc = crafter.craft(signal_orig, 0)

    signal_mono = crafted_doc["blob"]
    assert signal_mono.shape[0] == signal_orig.shape[1]
