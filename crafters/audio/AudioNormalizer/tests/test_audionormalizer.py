import numpy as np

from .. import AudioNormalizer


def test_audionormalizer():
    """here is my test code

    https://docs.pytest.org/en/stable/getting-started.html#create-your-first-test
    """
    signal_orig = np.random.randn(2, 31337)

    crafter = AudioNormalizer()
    crafted_docs = crafter.craft(np.stack([signal_orig, signal_orig]))

    assert len(crafted_docs) == 2
    signal_norm = crafted_docs[0]['blob']
    assert signal_norm.shape == signal_orig.shape
    assert np.min(signal_norm) == -1
    assert np.max(signal_norm) == 1

    signal_norm = crafted_docs[1]['blob']
    assert signal_norm.shape == signal_orig.shape
    assert np.min(signal_norm) == -1
    assert np.max(signal_norm) == 1
