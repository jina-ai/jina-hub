import os
import scipy

import numpy as np
from .. import TFIDFTextEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def print_array_info(x, x_varname):
    print('\n')
    print(f'type({x_varname})={type(x)}')
    print(f'{x_varname}.dtype={x.dtype}')
    print(f'{x_varname}.shape={x.shape}')


def test_tfidf_text_encoder():
    # Input
    text = np.array(['Han likes eating pizza'])

    # Encoder embedding 
    encoder = TFIDFTextEncoder()

    print_array_info(text, 'text')
    embeddeding = encoder.encode(text)
    print_array_info(embeddeding, 'embeddeding')

    # Compare with ouptut 
    expected = scipy.sparse.load_npz(os.path.join(cur_dir, 'expected.npz'))
    np.testing.assert_almost_equal(embeddeding.todense(), expected.todense(), decimal=4)


def test_tfidf_text_encoder_batch():
    # Input
    text_batch = np.array(['Han likes eating pizza', 'Han likes pizza', 'Jina rocks'])

    # Encoder embedding 
    encoder = TFIDFTextEncoder()

    print_array_info(text_batch, 'text_batch')
    embeddeding_batch = encoder.encode(text_batch)
    print_array_info(embeddeding_batch, 'embeddeding_batch')

    # Compare with ouptut 
    expected_batch = scipy.sparse.load_npz(os.path.join(cur_dir, 'expected_batch.npz'))
    np.testing.assert_almost_equal(embeddeding_batch.todense(), expected_batch.todense(), decimal=2)
