from .. import PDFTextExtractor



def test_io_uri():
    crafter = PDFTextExtractor()
    text = crafter.craft(uri='cats_are_awesome.pdf', buffer=None)
    print(text)


def test_io_buffer():
    pdf_obj = open('paper35.pdf', 'rb')
    crafter = PDFTextExtractor()
    text = crafter.craft(uri=None, buffer=pdf_obj)
    print(text)
