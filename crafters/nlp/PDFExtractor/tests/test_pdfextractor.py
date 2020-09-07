from .. import PDFTextExtractor

def create_pdf():
    from reportlab.pdfgen.canvas import Canvas
    canvas = Canvas("awesome_pdf_test.pdf")
    canvas.drawString(72, 72, "Cats rules")
    canvas.save()


def test_read_pdf():
    crafter = PDFTextExtractor()
