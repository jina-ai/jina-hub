from .. import DeepSegmenter


def test_deepsegmenter():
    crafter = DeepSegmenter()
    text = 'I am Batman i live in gotham'
    chunks = crafter.craft(text, 0)
    assert len(chunks) == 2
    assert chunks[0]['text'] == 'I am Batman'
    assert chunks[1]['text'] == 'i live in gotham'
