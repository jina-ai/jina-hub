# PDFExtractorSegmenter

## 🗝️ Key Concepts
This is a simple flow test where you can use the PDFExtractorSegmenter.
This Segmenter will create a chunk per blob.

So if we have a PDF with 2 images and text, we will have 3 chunks; one per each image and one per the text.


## 🏃️ Flow
First you define your flow here:
    
    f = Flow().add(uses='PDFExtractorSegmenter', array_in_pb=True)

If you have a PDF with images and text, the ```array_in_pb=True``` is necessary.

And then we search in: 

    with f:
        f.search(input_fn=search_generator(path=path, buffer=None), on_done=validate_mix_fn)

For this example I have 3 different PDF files to test. 

 - **Image only:** If we want to test the only-images PDF file, we use  ```validate_img_fn``` in the ```on_done```
 - **Text only:** If we want to test the only-text PDF file, we use  ```validate_text_fn``` in the ```on_done```
 - **Image and text:** If we want to test the mixed PDF file (images and text), we use  ```validate_mix_fn``` in the ```on_done```
 
 ## 🤔  Tests
 
For this examples we have 2 images per pdf, one of size 1024 x 660, and the other of 1191 x 626.

 #### 🧪 Images
  
For the image validations we will check each chunk and compare it with the original image

    assert blob.shape[1], blob.shape[0] == img.size

 #### 🧪 Text
And for the text we will check that that chunk corresponds with the expected text

```
expected_text = A cat poem I love cats, I love every kind of cat, I just wanna hug all of them, but I can't,
I'm thinking about cats again I think about how cute they are\nAnd their whiskers and their nose"

assert expected_text == d.chunks[2].text
```