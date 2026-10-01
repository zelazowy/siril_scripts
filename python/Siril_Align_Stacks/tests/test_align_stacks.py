import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from skimage.transform import SimilarityTransform

from Siril_Align_Stacks import align, largest_rectangle, preview_pixels, save_outputs, warp


class AlignmentTests(unittest.TestCase):
    def test_preview_bright_stars_do_not_hide_rgb_background(self):
        rng = np.random.default_rng(9)
        image = rng.normal(.01, .00002, (100, 120)).astype(np.float32)
        image[::10, ::10] = 1
        rgb = np.stack([image, image*1.2, image*.8])
        before = rgb.copy()
        preview = preview_pixels(rgb, np.ones(image.shape, bool))
        self.assertAlmostEqual(float(np.median(preview)), 63, delta=1)
        self.assertGreater(int(preview[0, 0]), 250)
        np.testing.assert_array_equal(rgb, before)
        constant = preview_pixels(np.ones((10, 10)), np.ones((10, 10), bool))
        self.assertTrue((constant == 0).all())

    def test_preview_ignores_padding_and_preserves_data(self):
        rng = np.random.default_rng(7)
        image = rng.normal(.1, .001, (100, 120)).astype(np.float32)
        common = np.zeros(image.shape, bool)
        common[20:80, 20:100] = True
        padded = image.copy()
        padded[~common] = 0
        before = padded.copy()
        expected = preview_pixels(image, common)
        actual = preview_pixels(padded, common)
        np.testing.assert_array_equal(actual[common], expected[common])
        np.testing.assert_array_equal(padded, before)
        self.assertLess(np.median(actual[common]), 230)
        self.assertTrue((actual[~common] == 0).all())
        rgb = np.stack([padded, padded, padded])
        np.testing.assert_allclose(preview_pixels(rgb, common), actual, atol=1)

    def test_rectangle_against_brute_force(self):
        rng = np.random.default_rng(123)
        for _ in range(30):
            mask = rng.random((5, 6)) > .3
            x, y, w, h = largest_rectangle(mask)
            self.assertTrue(mask[y:y+h, x:x+w].all())
            maximum = max((b-a)*(d-c) for a in range(5) for b in range(a+1, 6)
                          for c in range(6) for d in range(c+1, 7)
                          if mask[a:b, c:d].all())
            self.assertEqual(w*h, maximum)

    def test_mixed_rgb_mono_registration_and_save(self):
        rng = np.random.default_rng(42)
        stars = np.zeros((256, 280), np.float32)
        for y, x in rng.integers([25, 25], [230, 255], size=(60, 2)):
            stars[y, x] += rng.uniform(10, 50)
        reference = gaussian_filter(stars, 1.5) + .015
        rgb = np.stack([reference, reference * 2, reference * .5])
        transform = SimilarityTransform(scale=1.01, rotation=.025, translation=(6, -4))
        moving, _ = warp(reference, transform.params, reference.shape)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/'rgb.fit', Path(directory)/'ha.fit']
            header = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                                  'CRPIX1': 140., 'CRPIX2': 128., 'CRVAL1': 10.,
                                  'CRVAL2': 20., 'CDELT1': -.001, 'CDELT2': .001})
            fits.writeto(paths[0], rgb, header)
            fits.writeto(paths[1], moving, fits.Header({'FILTER': 'Ha', 'CRPIX1': 999.}))
            images, common = align(paths)
            crop = largest_rectangle(common)
            x, y, w, h = crop
            error = np.sqrt(np.mean((images[1][0][y:y+h, x:x+w]-reference[y:y+h, x:x+w])**2))
            self.assertLess(error, .025)
            outputs = save_outputs(paths, images, crop, '_aligned')
            with fits.open(outputs[0]) as hdus:
                self.assertEqual(hdus[0].data.shape, (3, h, w))
                np.testing.assert_array_equal(hdus[0].data, rgb[:, y:y+h, x:x+w])
            with fits.open(outputs[1]) as hdus:
                self.assertEqual(hdus[0].data.shape, (h, w))
                self.assertEqual(hdus[0].header['FILTER'], 'Ha')
                self.assertEqual(hdus[0].header['CRPIX1'], 140-x)
                self.assertEqual(hdus[0].header['CRPIX2'], 128-y)
            with self.assertRaises(ValueError):
                save_outputs(paths, images, crop, '_aligned')
            np.testing.assert_array_equal(fits.getdata(paths[0]), rgb)


if __name__ == '__main__':
    unittest.main()
