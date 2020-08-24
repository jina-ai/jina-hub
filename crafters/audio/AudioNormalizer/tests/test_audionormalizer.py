import numpy as np

from .. import AudioNormalizer


def test_audionormalizer():
    """here is my test code

    https://docs.pytest.org/en/stable/getting-started.html#create-your-first-test
    """
    signal_orig = np.random.randn(2, 31337)

    crafter = AudioNormalizer()
    crafted_doc = crafter.craft(signal_orig, 0)

    signal_norm = crafted_doc["blob"]
    assert signal_norm.shape == signal_orig.shape
    assert np.min(signal_norm) == -1
    assert np.max(signal_norm) == 1
