from .. import PDFTextExtractor



def test_io_uri():
    crafter = PDFTextExtractor()
    text = crafter.craft(uri='cats_are_awesome.pdf', buffer=None)
    print(text)


def test_io_buffer():
    import requests
    url = 'http://ceur-ws.org/Vol-2410/paper35.pdf'
    response = requests.get(url)

    crafter = PDFTextExtractor()
    text = crafter.craft(uri=None, buffer=response.content)
    print(text)
