import os
import pytest
import spacy
import numpy as np

from .. import SpacyTextEncoder


cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def multilingual_model_name():
    return 'en_core_web_sm'


def print_array_info(x, x_varname):
    print('\n')
    print(f'type({x_varname})={type(x)}')
    print(f'{x_varname}.dtype={x.dtype}')
    print(f'len({x_varname})={len(x)}')
    print(f'{x_varname}.shape={x.shape}')


def test_spacy_available_models():
    models = ['en_core_web_sm']
    for model in models:
        _ = spacy.load(model)


def test_spacy_text_encoder():
    # Input
    text = np.array(['Han likes eating pizza'])

    # Encoder embedding
    encoder = SpacyTextEncoder()
    print_array_info(text, 'text')
    embeddeding_np = encoder.encode(text)
    print_array_info(embeddeding_np, 'embeddeding_np')

    # Compare with ouptut
    expected = np.load(os.path.join(cur_dir, 'expected_data.npy'))
    np.testing.assert_almost_equal(embeddeding_np, expected, decimal=4)


def test_spacy_text_encoder_batch():
    # Input
    text_batch = np.array(['Han likes eating pizza', 'Han likes pizza', 'Jina rocks'])

    # Encoder embedding
    encoder = SpacyTextEncoder()
    print_array_info(text_batch, 'text_batch')
    embeddeding_batch_np = encoder.encode(text_batch)
    print_array_info(embeddeding_batch_np, 'embeddeding_batch_np')

    # Compare with ouptut
    expected_batch = np.load(os.path.join(cur_dir, 'expected_batch_data.npy'))
    np.testing.assert_almost_equal(embeddeding_batch_np, expected_batch, decimal=4)


def test_unsupported_lang(tmp_path):
    dummy1 = spacy.blank('xx')
    dummy1_dir_path = tmp_path / 'xx1'
    dummy1.to_disk(dummy1_dir_path)
    dummy2 = spacy.blank('xx')
    dummy2_dir_path = tmp_path / 'xx2'
    dummy2.to_disk(dummy2_dir_path)
    # No available language
    with pytest.raises(IOError):
        SpacyTextEncoder('abcd')

    # Language does not have DependencyParser should thrown an error
    # when try to use default encoder
    with pytest.raises(ValueError):
        SpacyTextEncoder(dummy1_dir_path, use_default_encoder=True)

    # And should be fine when 'parser' pipeline is added
    dummy1.add_pipe('parser')
    dummy1.to_disk(dummy1_dir_path)
    SpacyTextEncoder(dummy1_dir_path, use_default_encoder=True)

    # Language does not have SentenceRecognizer should thrown an error
    # when try to use non default encoder
    with pytest.raises(ValueError):
        SpacyTextEncoder(dummy2_dir_path, use_default_encoder=False)

    # And should be fine when 'senter' pipeline is added
    dummy2.add_pipe('tok2vec')
    dummy2.to_disk(dummy2_dir_path)
    SpacyTextEncoder(dummy2_dir_path, use_default_encoder=False)
