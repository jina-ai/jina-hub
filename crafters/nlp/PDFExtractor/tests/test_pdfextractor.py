from .. import PDFTextExtractor

def create_pdf():
    from reportlab.pdfgen.canvas import Canvas
    canvas = Canvas('awesome_name.pdf')
    canvas.drawString(20, 500, "Lorem ipsum dolor sit amet, "
                              "consectetur adipiscing elit. "
                              "Cras vehicula facilisis erat iaculis facilisis. "
                              "Sed aliquam mi in erat tincidunt, in posuere ex lobortis. "
                              "Mauris sit amet nunc nec sapien pellentesque mattis. "
                              "Vivamus sit amet quam diam. Pellentesque vitae ullamcorper nunc. "
                              "Nulla nec malesuada lectus. Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                              "Aliquam malesuada, lorem vel gravida rutrum, odio dui pellentesque enim, "
                              "sed faucibus nisl leo ut massa. "
                              "In posuere auctor pellentesque. Donec cursus ipsum ut odio dictum ultricies. "
                              "Suspendisse volutpat accumsan imperdiet. Pellentesque sodales pharetra egestas. S"
                              "ed tincidunt nec eros vel mollis. Etiam blandit leo vitae mattis bibendum. "
                              "Nunc rutrum hendrerit ligula. ")
    canvas.save()

def test_read_pdf():
    crafter = PDFTextExtractor()
    text = crafter.craft('cats_are_awesome.pdf')
    print(text)

