from .. import PDFTextExtractor

def test_create_pdf():
    from reportlab.pdfgen.canvas import Canvas
    canvas = Canvas("awesome_pdf_test.pdf")
    canvas.drawString(72, 72, "Cats rules")
    canvas.save()


