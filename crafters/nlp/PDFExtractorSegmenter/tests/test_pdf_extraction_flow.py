from jina.flow import Flow
import os

def test_pdf_flow():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'cats_are_awesome.pdf')

    f = Flow().add(uses='PDFExtractorSegmenter')
    with f:
        f.index_files([path], output_fn=print)

