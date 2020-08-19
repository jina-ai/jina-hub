from . import JinaImageTestCase
from .. import FiveImageCropper


class ImageSegmentTestCase(JinaImageTestCase):
    def test_five_image_crop(self):
        img_size = 217
        img_array = self.create_random_img_array(img_size, img_size)
        output_dim = 20
        crafter = FiveImageCropper(output_dim)
        chunks = crafter.craft(img_array)
        self.assertEqual(len(chunks), 5)
        self.assertEqual(chunks[0]['location'], (0, 0))
        self.assertEqual(chunks[1]['location'], (0, 197))
        self.assertEqual(chunks[2]['location'], (197, 0))
        self.assertEqual(chunks[3]['location'], (197, 197))
        self.assertEqual(chunks[4]['location'], (98, 98))
