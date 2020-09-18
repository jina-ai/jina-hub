from jina.flow import Flow
import os


def print_res(resp):
    for d in resp.search.docs:
        print(f'Ta-Dah🔮, here is what we found:')
        print('👉%s' % d.chunks[0].text)


def test_pdf_flow():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'cats_are_awesome_text.pdf')

    f = Flow().add(uses='PDFExtractorSegmenter')
    with f:
        f.index_files([path])

    with f:
        f.search([path], output_fn=print_res)

