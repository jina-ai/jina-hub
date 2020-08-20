from . import JinaImageTestCase
from .. import RandomImageCropper


class ImageSegmentTestCase(JinaImageTestCase):
    def test_random_crop(self):
        img_size = 217
        img_array = self.create_random_img_array(img_size, img_size)
        output_dim = 20
        num_patches = 20
        crafter = RandomImageCropper(output_dim, num_patches)
        chunks = crafter.craft(img_array)
        assert len(chunks) == num_patches
        for chunk in chunks:
            self.assertTrue(chunk['location'][0] <= (img_size - output_dim))
            self.assertTrue(chunk['location'][1] <= (img_size - output_dim))
