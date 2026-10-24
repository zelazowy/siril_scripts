import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, Sip
from Siril_Align_Stacks import (validate_sip, refine_inverse_sip, same_output_grid,
    run_siril, SIPOptions, align_sip, centroid_residuals, stellar_centroids,
    largest_rectangle, save_outputs, canonical_sip_header, astap_solution_header,
    solve_astap_stack)


def solution(coeff=5e-7):
    w = WCS(naxis=2)
    w.wcs.crpix = [129, 129]
    w.wcs.cd = np.diag([-.0005, .0005])
    w.wcs.crval = [315.4, 68.16]
    w.wcs.ctype = ['RA---TAN-SIP', 'DEC--TAN-SIP']
    a = np.zeros((4, 4)); b = a.copy()
    a[3, 0] = coeff; b[0, 3] = -coeff
    w.sip = Sip(a, b, -a, -b, w.wcs.crpix)
    return w


class SIPTests(unittest.TestCase):
    def test_astap_lower_terms_preserve_world_mapping(self):
        w = solution()
        a, b = w.sip.a.copy(), w.sip.b.copy()
        a[0, 0], a[1, 0], b[0, 0], b[0, 1] = .12, .001, -.08, -.002
        w.sip = Sip(a, b, None, None, w.wcs.crpix)
        header, changed = canonical_sip_header(w.to_header(relax=True), (256, 256))
        self.assertTrue(changed)
        validate_sip(header, (256, 256), 3)
        points = np.array([[0., 0.], [200., 110.], [255., 255.]])
        np.testing.assert_allclose(WCS(header).all_pix2world(points, 0),
                                   w.all_pix2world(points, 0), atol=1e-9)

    def test_astap_linear_solution_cannot_inherit_input_sip(self):
        old = solution().to_header(relax=True)
        linear = solution(); linear.sip = None
        linear.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        with self.assertRaisesRegex(ValueError, 'linear WCS'):
            astap_solution_header(old, linear.to_header(), (256, 256))

    def test_shared_solver_transfers_only_verified_astrometry(self):
        from astro_platesolve import SolveResult
        data=np.arange(256*256,dtype=np.float32).reshape(256,256)/10000
        before=data.copy();header=solution().to_header(relax=True);header['EXPTIME']=123.
        with tempfile.TemporaryDirectory() as folder:
            with patch('astro_platesolve.solve_image',return_value=SolveResult(solution().to_header(relax=True),'test')) as run:
                accepted=solve_astap_stack(data,header,SIPOptions(),Path(folder)/'solver.fit',lambda _:None)
        self.assertTrue(run.call_args.args[2].require_sip)
        validate_sip(accepted,data.shape,3)
        self.assertEqual(accepted['EXPTIME'],123.)
        self.assertEqual(accepted['ALNBACK'],'test')
        np.testing.assert_array_equal(data,before)

    def test_reject_linear_forward_only_and_wrong_order(self):
        header = solution().to_header(relax=True)
        validate_sip(header, (256, 256), 3)
        for prefix in ('AP', 'BP'):
            for key in list(header):
                if key.startswith(prefix+'_'):
                    del header[key]
        with self.assertRaisesRegex(ValueError, 'inverse SIP'):
            validate_sip(header, (256, 256), 3)
        with self.assertRaises(ValueError):
            validate_sip(solution().to_header(relax=True), (256, 256), 2)
        w = solution(); w.sip = None
        with self.assertRaises(ValueError):
            validate_sip(w.to_header(), (256, 256), 3)

    def test_inverse_refinement_retains_independent_forward_models(self):
        for coefficient in (3e-6, -4e-6):
            header = solution(coefficient).to_header(relax=True)
            with self.assertRaisesRegex(ValueError, 'reverse polynomials differ'):
                validate_sip(header, (256, 256), 3)
            refined, error = refine_inverse_sip(header, (256, 256))
            self.assertLess(error, .05)
            self.assertEqual(refined['A_3_0'], header['A_3_0'])
            self.assertEqual(refined['B_0_3'], header['B_0_3'])
            validate_sip(refined, (256, 256), 3)

    def test_zero_exit_failed_batch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            options = SIPOptions('siril-cli', str(Path(folder)/'config.ini'))
            with patch('subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout='log: Script execution failed.')):
                with self.assertRaisesRegex(RuntimeError, 'batch failed'):
                    run_siril(options, folder, ['invalid'])

    def test_grid_mismatch_rejected(self):
        w = solution(); w.sip = None; w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        header = w.to_header()
        same_output_grid([header, header], (256, 256))
        shifted = header.copy(); shifted['CRPIX1'] += 1
        with self.assertRaises(ValueError):
            same_output_grid([header, shifted], (256, 256))

    @unittest.skipUnless(os.environ.get('SIRIL_TEST_CONFIG'), 'native integration opt-in')
    def test_native_independent_sip_rgb_and_mono(self):
        rng = np.random.default_rng(8)
        sky = solution().wcs_pix2world(rng.uniform(25, 230, (100, 2)), 0)
        yy, xx = np.mgrid[:256, :256]
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/'rgb.fit', Path(folder)/'mono.fit']
            originals = []
            for i, coeff in enumerate((3e-6, -1e-6)):
                w = solution(coeff)
                if i == 0:
                    a, b = w.sip.a.copy(), w.sip.b.copy()
                    a[0, 0], a[1, 0], b[0, 0], b[0, 1] = .12, .001, -.08, -.002
                    w.sip = Sip(a, b, -a, -b, w.wcs.crpix)
                pixels = w.all_world2pix(sky, 0)
                data = np.full((256, 256), .02, np.float32)
                for j, (x, y) in enumerate(pixels):
                    data += (.1+.01*(j % 10))*np.exp(-((xx-x)**2+(yy-y)**2)/4)
                data += rng.normal(0, .00002, data.shape).astype(np.float32)
                if i == 0: data = np.stack([data, data*1.2, data*.8])
                header = w.to_header(relax=True)
                fits.writeto(paths[i], data, header)
                originals.append(data.copy())
            options = SIPOptions(os.environ['SIRIL_TEST_CLI'],
                                 os.environ['SIRIL_TEST_CONFIG'], solver='existing')
            images, mask = align_sip(paths, options)
            self.assertEqual(images[0][0].shape[0], 3)
            self.assertEqual(images[1][0].ndim, 2)
            np.testing.assert_allclose(images[0][0][1], images[0][0][0]*1.2, atol=1e-6)
            np.testing.assert_allclose(images[0][0][2], images[0][0][0]*.8, atol=1e-6)
            self.assertFalse(np.array_equal(images[0][0], originals[0]))
            stats = centroid_residuals(stellar_centroids(images[0][0]),
                                      stellar_centroids(images[1][0]), mask.shape, mask)
            self.assertGreater(stats['outer']['count'], 10)
            self.assertLess(stats['outer']['rms'], .1)
            self.assertAlmostEqual(float(np.median(images[1][0][mask])), .02, delta=.0001)
            crop = largest_rectangle(mask)
            outputs = save_outputs(paths, images, crop, '_aligned_sip')
            headers = [fits.getheader(p) for p in outputs]
            same_output_grid(headers, (crop[3], crop[2]))
            for i, path in enumerate(paths):
                np.testing.assert_array_equal(fits.getdata(path), originals[i])
                self.assertEqual(headers[i]['ALNMODE'], 'SIP')


if __name__ == '__main__':
    unittest.main()
