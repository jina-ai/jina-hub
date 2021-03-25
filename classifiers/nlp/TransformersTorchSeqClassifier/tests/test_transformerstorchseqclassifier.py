from .. import TransformersTorchSeqClassifier
import numpy as np
import pytest


def test_sequencetransformerstorchclassifier():
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    classifier = TransformersTorchSeqClassifier(model_name)
    data = np.stack(
        [
            'Today is a good day.',
            "Can't wait for tomorrow!",
            "Today is a good day. Can't wait for tomorrow!",
        ]
    )
    output = classifier.predict(data)
    assert output.shape[0] == data.shape[0]
    assert output.shape[1] == classifier.model.config.num_labels
    assert (output.argmax(axis=1) == 1).all()


@pytest.mark.parametrize(
    'data',
    [
        (
            np.stack(
                [
                    "Aujourd'hui est un bon jour.",
                    'Je ne peux pas attendre demain!',
                    "Aujourd'hui est un bon jour. Je ne peux pas attendre demain!",
                ]
            )
        ),
        (
            np.stack(
                [
                    'Heute ist ein guter Tag.',
                    'Ich kann nicht auf morgen warten!',
                    'Heute ist ein guter Tag. Ich kann nicht auf morgen warten!',
                ]
            )
        ),
    ],
)
def test_multilingual(data):
    model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
    classifier = TransformersTorchSeqClassifier(model_name)
    output = classifier.predict(data)
    assert output.shape[0] == data.shape[0]
    assert output.shape[1] == 5
    # all predictions should be more positive than negative, out of 5
    assert (output.argmax(axis=1) > 2).all()
    assert classifier.model.config.id2label[output[2, :].argmax()] == '5 stars'
