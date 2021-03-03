import numpy as np

from .. import AudioMonophoner


def test_audiomonophoner():
    """here is my test code

    https://docs.pytest.org/en/stable/getting-started.html#create-your-first-test
    """
    signal_orig = np.random.randn(2, 31337)

    crafter = AudioMonophoner()
    crafted_docs = crafter.craft([signal_orig, signal_orig])

    assert len(crafted_docs) == 2

    signal_mono = crafted_docs[0]['blob']
    assert signal_mono.shape[0] == signal_orig.shape[1]

    signal_mono = crafted_docs[1]['blob']
    assert signal_mono.shape[0] == signal_orig.shape[1]
