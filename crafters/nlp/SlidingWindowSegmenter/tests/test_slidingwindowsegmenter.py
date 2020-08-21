from .. import SlidingWindowSegmenter


def test_sliding_window_segmenter():
    window_size = 20
    step_size = 10
    sliding_window_segmenter = SlidingWindowSegmenter(
        window_size=window_size, step_size=step_size)
    text = 'It is a sunny day!!!! When Andy comes back, we are going to the zoo.'
    crafted_chunk_list = sliding_window_segmenter.craft(text, 0)
    assert len(crafted_chunk_list) == len(text) // step_size
