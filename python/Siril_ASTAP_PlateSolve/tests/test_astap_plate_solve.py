"""Regression tests with a real Qt subprocess and a simulated Siril image API."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from contextlib import nullcontext
from pathlib import Path
import shlex
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PyQt6 import QtCore, QtWidgets
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Siril_ASTAP_PlateSolve as solver


def header_text(header):
    return header.tostring(sep="\n", endcard=True, padding=False)


class FakeSiril:
    def __init__(self, root, pixels=None, header=None):
        self.root = Path(root)
        self.filename = str(self.root / "original.tiff")
        self.pixels = pixels if pixels is not None else np.arange(3*64*80, dtype=np.uint16).reshape(3, 64, 80)
        self.header = header_text(header if header is not None else fits.PrimaryHDU(self.pixels).header)
        self.writes = []
        self.applied_headers = []
        self.messages = []

    def get_siril_userdatadir(self):
        return str(self.root)

    def is_image_loaded(self):
        return True

    def get_image_filename(self):
        return self.filename

    def get_image_pixeldata(self):
        return self.pixels

    def get_image_fits_header(self):
        return self.header

    def log(self, text):
        self.messages.append(text)

    def save_image_file(self, pixels, header, filename):
        self.writes.append((Path(filename), np.array(pixels, copy=True)))
        parsed = fits.Header.fromstring(header, sep="\n")
        fits.writeto(filename, pixels, parsed, overwrite=True)
        return True

    def image_lock(self):
        return nullcontext()

    def undo_save_state(self, label):
        pass

    def set_image_metadata_from_header_string(self, header):
        self.header = header
        self.applied_headers.append(header)

    def set_image_filename(self, name):
        self.filename = name

    def pix2radec(self, x, y):
        return WCS(fits.Header.fromstring(self.header, sep="\n"), naxis=2).all_pix2world([[x, y]], 0)[0]


class AstapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.siril = FakeSiril(self.root)
        self.dialog = solver.AstapPlateSolveDialog(self.siril)
        self.dialog.path_edit.setText('/bin/sh')
        self.dialog.correct_mirror_check.setChecked(False)
        self.wcs_header = fits.Header({
            'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
            'CRPIX1': 40., 'CRPIX2': 32., 'CRVAL1': 313., 'CRVAL2': 60.,
            'CD1_1': -.00057, 'CD1_2': 0., 'CD2_1': 0., 'CD2_2': .00057,
            'EQUINOX': 2000.,
        })

    def tearDown(self):
        self.dialog.close()
        self.app.processEvents()
        self.temp.cleanup()

    def solved_result(self, *args):
        args[1].with_suffix('.wcs').write_text(header_text(self.wcs_header))
        return {'PLTSOLVD': 'T'}, ''

    def fake_executable(self, code):
        script = self.root / 'fake_astap.py'
        script.write_text(code)
        executable = self.root / 'fake_astap'
        executable.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(script)) + ' "$@"\n')
        executable.chmod(0o755)
        return executable

    def test_undefined_equinox_and_header_terminator(self):
        original = fits.Header({'EQUINOX': 9.87654321e107, 'OBJECT': 'Barnard 150'})
        text = header_text(original) + '\x00'
        clean = solver.prepare_solver_header(text)
        parsed = fits.Header.fromstring(clean, sep='\n')
        self.assertEqual(parsed['EQUINOX'], 2000.)
        self.assertEqual(parsed['OBJECT'], 'Barnard 150')
        self.assertEqual(original['EQUINOX'], 9.87654321e107)
        self.assertNotIn('\x00', clean)
        for value in (1950., 2000., 2050.):
            valid = header_text(fits.Header({'EQUINOX': value}))
            self.assertEqual(solver.prepare_solver_header(valid), valid)
        # Fixed-width shared memory headers also work.
        fixed = original.tostring(endcard=True, padding=False)
        self.assertEqual(fits.Header.fromstring(solver.prepare_solver_header(fixed), sep='\n')['EQUINOX'], 2000.)

    def test_compact_proxy_preserves_peaks_grid_and_source(self):
        yy, xx = np.mgrid[:220, :240]
        mono = (.1 + .0002*xx + .0001*yy).astype(np.float32)
        centers = [(y, x) for y in range(20, 201, 30) for x in range(20, 221, 30)]
        for y, x in centers:
            mono += (.4*np.exp(-((yy-y)**2+(xx-x)**2)/4)).astype(np.float32)
        before = mono.copy()
        proxy = solver.prepare_compact_star_pixels(mono)
        np.testing.assert_array_equal(mono, before)
        self.assertEqual(proxy.shape, mono.shape)
        self.assertEqual(proxy.dtype, np.float32)
        self.assertTrue(np.isfinite(proxy).all())
        self.assertAlmostEqual(float(proxy.max()), 65535., delta=.01)
        for y, x in centers:
            peak = np.unravel_index(proxy[y-3:y+4, x-3:x+4].argmax(), (7, 7))
            self.assertEqual(peak, (3, 3))
        self.assertIsNone(solver.prepare_compact_star_pixels(np.ones((64, 80), dtype=np.float32)))

    def test_merge_rejects_stale_pc_and_sip_carried_by_astap(self):
        stale = self.wcs_header.copy()
        stale['PC1_1'] = -1.
        stale['PC2_2'] = -1.
        stale['A_ORDER'] = 2
        stale['A_2_0'] = .001
        stale['B_ORDER'] = 2
        stale['B_0_2'] = .001
        text = solver.merge_astrometry(self.siril.header, header_text(stale))
        merged = fits.Header.fromstring(text, sep='\n')
        self.assertNotIn('PC1_1', merged)
        self.assertNotIn('A_ORDER', merged)
        self.assertNotIn('B_0_2', merged)
        self.assertIsNone(WCS(merged, naxis=2).sip)
        np.testing.assert_allclose(WCS(merged, naxis=2).pixel_scale_matrix,
                                   [[-.00057, 0], [0, .00057]])
        stale['CTYPE1'] = 'RA---TAN-SIP'
        stale['CTYPE2'] = 'DEC--TAN-SIP'
        merged = fits.Header.fromstring(solver.merge_astrometry(self.siril.header, header_text(stale)), sep='\n')
        self.assertEqual(merged['A_2_0'], .001)
        self.assertIsNotNone(WCS(merged, naxis=2).sip)

    def jpeg(self, suffix='.jpg', embedded=None, orientation=1, size=(80, 64), description=None):
        path = self.root / ('photo' + suffix)
        exif = Image.Exif()
        exif[274] = orientation
        if embedded is not None:
            exif[270] = header_text(embedded)
        elif description is not None:
            exif[270] = description
        Image.new('RGB', size, (40, 50, 60)).save(path, exif=exif)
        return path

    def embedded_jpeg_header(self):
        embedded = fits.PrimaryHDU(self.siril.pixels).header
        embedded.update(self.wcs_header)
        embedded['FOCALLEN'] = 653.
        embedded['XPIXSZ'] = 6.54
        embedded['YPIXSZ'] = 6.54
        embedded['EQUINOX'] = 9.87654321e107
        return embedded

    def test_jpeg_exif_guides_fresh_solve_without_replacing_geometry(self):
        embedded = self.embedded_jpeg_header()
        for ending in ('\n', '\r\n', '\r'):
            path = self.jpeg(description=header_text(embedded).replace('\n', ending))
            current = fits.Header.fromstring(self.siril.header, sep='\n')
            current['EQUINOX'] = 9.87654321e107
            current['RA'] = 9.87654321e107
            current['OBJECT'] = 'Current edited metadata'
            actual, message = solver.prepare_jpeg_solver_header(path, header_text(current), 80, 64)
            hints = fits.Header.fromstring(actual, sep='\n')
            self.assertEqual(hints['CRVAL1'], 313.)
            self.assertEqual(hints['FOCALLEN'], 653.)
            self.assertEqual(hints['EQUINOX'], 2000.)
            self.assertNotIn('RA', hints)
            self.assertEqual(hints['OBJECT'], 'Current edited metadata')
            self.assertEqual(hints['NAXIS3'], 3)
            self.assertIn('fresh ASTAP solve', message)

    def test_jpeg_ignores_stale_rotated_missing_and_invalid_exif(self):
        embedded = self.embedded_jpeg_header()
        singular = embedded.copy()
        singular['CD1_1'] = 0.
        singular['CD2_2'] = 0.
        malformed = '\n'.join('CRVAL1  = ?????'.ljust(80) if line.startswith('CRVAL1') else line
                              for line in header_text(embedded).splitlines())
        variants = [({}, (80, 64)), ({'description': 'ordinary caption'}, (80, 64)),
                    ({'description': malformed}, (80, 64)),
                    ({'embedded': embedded, 'size': (40, 32)}, (40, 32)),
                    ({'embedded': embedded, 'orientation': 6}, (80, 64)),
                    ({'embedded': embedded}, (60, 40)),
                    ({'embedded': singular}, (80, 64))]
        for kwargs, (width, height) in variants:
            with self.subTest(kwargs=kwargs):
                path = self.jpeg(**kwargs)
                actual, _ = solver.prepare_jpeg_solver_header(path, self.siril.header, width, height)
                self.assertEqual(actual, self.siril.header)

    def test_jpeg_current_complete_wcs_has_priority(self):
        embedded = self.embedded_jpeg_header()
        path = self.jpeg(embedded=embedded)
        current = fits.Header.fromstring(self.siril.header, sep='\n')
        current.update(self.wcs_header)
        current['CRVAL1'] = 120.
        current['FOCALLEN'] = 800.
        actual, _ = solver.prepare_jpeg_solver_header(path, header_text(current), 80, 64)
        hints = fits.Header.fromstring(actual, sep='\n')
        self.assertEqual(hints['CRVAL1'], 120.)
        self.assertEqual(hints['FOCALLEN'], 800.)

    def test_jpeg_source_saves_lossless_fits_and_always_runs_astap(self):
        for suffix in ('.jpg', '.jpeg', '.JPG'):
            with self.subTest(suffix=suffix):
                source = self.jpeg(suffix=suffix, embedded=self.embedded_jpeg_header())
                original = source.read_bytes()
                self.siril.filename = str(source)
                self.siril.header = header_text(fits.PrimaryHDU(self.siril.pixels).header)
                with patch.object(self.dialog, '_run_astap', side_effect=self.solved_result) as run, patch.object(QtWidgets.QMessageBox, 'information') as complete, patch.object(QtWidgets.QMessageBox, 'critical') as failed:
                    self.dialog._solve_current_image()
                failed.assert_not_called()
                complete.assert_called_once()
                run.assert_called_once()
                output = source.with_name('photo_astap_solved.fit')
                np.testing.assert_array_equal(fits.getdata(output), self.siril.pixels)
                self.assertEqual(source.read_bytes(), original)
                self.assertIn('JPEG EXIF', self.dialog.log_box.toPlainText())

    def test_jpeg_without_embedded_header_still_solves(self):
        source = self.jpeg()
        self.siril.filename = str(source)
        original = source.read_bytes()
        with patch.object(self.dialog, '_run_astap', side_effect=self.solved_result) as run, patch.object(QtWidgets.QMessageBox, 'information'), patch.object(QtWidgets.QMessageBox, 'critical') as failed:
            self.dialog._solve_current_image()
        failed.assert_not_called()
        run.assert_called_once()
        self.assertIn('no embedded FITS header', self.dialog.log_box.toPlainText())
        self.assertEqual(source.read_bytes(), original)

    def test_cancel_during_proxy_preparation(self):
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            if checks == 3:
                raise solver.SolveCancelled('cancelled')
        with self.assertRaises(solver.SolveCancelled):
            solver.prepare_compact_star_pixels(np.ones((64, 80), dtype=np.float32), cancel)

    def test_normal_solve_does_not_build_compact_proxy(self):
        mono = solver.prepare_solver_pixels(self.siril.pixels)
        with patch.object(self.dialog, '_run_astap', side_effect=self.solved_result) as run, patch.object(solver, 'prepare_compact_star_pixels') as compact:
            self.dialog._solve_with_strategies(Path('/fake'), self.root/'working.fit', mono, self.siril.header)
        compact.assert_not_called()
        self.assertEqual(run.call_count, 1)

    def test_processed_recovery_precedes_wide_search_and_preserves_rgb(self):
        source = Path(self.siril.filename)
        source.write_bytes(b'original TIFF remains unchanged')
        before = self.siril.pixels.copy()
        def attempt(*args):
            if args[2] != 'Processed-image recovery':
                return {'PLTSOLVD': 'F', 'ERROR': 'No matching stars'}, ''
            return self.solved_result(*args)
        proxy = np.full((64, 80), 123., dtype=np.float32)
        with patch.object(self.dialog, '_run_astap', side_effect=attempt) as run, patch.object(solver, 'prepare_compact_star_pixels', return_value=proxy), patch.object(QtWidgets.QMessageBox, 'information') as complete, patch.object(QtWidgets.QMessageBox, 'critical') as failed:
            self.dialog._solve_current_image()
        failed.assert_not_called()
        complete.assert_called_once()
        self.assertEqual([c.args[3] for c in run.call_args_list], [30, 30, 30])
        output = source.with_name('original_astap_solved.fit')
        np.testing.assert_array_equal(fits.getdata(output), before)
        np.testing.assert_array_equal(self.siril.pixels, before)
        np.testing.assert_array_equal(self.siril.writes[-2][1], proxy)
        self.assertEqual(fits.getheader(output)['NAXIS3'], 3)
        self.assertEqual(source.read_bytes(), b'original TIFF remains unchanged')
        self.assertTrue(self.dialog.solve_button.isEnabled())
        self.assertFalse(self.dialog.cancel_button.isEnabled())

    def test_insufficient_stars_and_timeout_still_allow_recovery(self):
        for failure in ({'PLTSOLVD': 'F', 'ERROR': 'Not enough stars detected.'}, solver.AstapTimeout('time limit')):
            with self.subTest(failure=failure):
                def attempt(*args):
                    if args[2] == 'Processed-image recovery':
                        return self.solved_result(*args)
                    if isinstance(failure, Exception):
                        raise failure
                    return failure, ''
                with patch.object(self.dialog, '_run_astap', side_effect=attempt), patch.object(solver, 'prepare_compact_star_pixels', return_value=np.ones((64,80), dtype=np.float32)):
                    result, _ = self.dialog._solve_with_strategies(Path('/fake'), self.root/'working.fit', solver.prepare_solver_pixels(self.siril.pixels), self.siril.header)
                self.assertEqual(result['PLTSOLVD'], 'T')

    def test_cancel_button_kills_process_prevents_save_and_allows_restart(self):
        pidfile = self.root/'pid'
        executable = self.fake_executable(f'''import os,time
from pathlib import Path
Path({str(pidfile)!r}).write_text(str(os.getpid()))
print('Still solving', flush=True)
while True: time.sleep(.05)
''')
        source = Path(self.siril.filename)
        source.write_bytes(b'unchanged')
        self.dialog.path_edit.setText(str(executable))
        timer = QtCore.QTimer()
        def click_when_started():
            if pidfile.exists():
                timer.stop()
                self.dialog.cancel_button.click()
        timer.timeout.connect(click_when_started)
        timer.start(20)
        start = time.monotonic()
        with patch.object(QtWidgets.QMessageBox, 'critical') as failed, patch.object(QtWidgets.QMessageBox, 'information') as complete:
            try:
                self.dialog._solve_current_image()
            finally:
                timer.stop()
        self.assertLess(time.monotonic()-start, 3)
        failed.assert_not_called()
        complete.assert_not_called()
        self.assertEqual(self.siril.applied_headers, [])
        self.assertFalse(source.with_name('original_astap_solved.fit').exists())
        self.assertEqual(source.read_bytes(), b'unchanged')
        self.assertIsNone(self.dialog._astap_process)
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)
        self.assertIn('cancelled', self.dialog.log_box.toPlainText())
        self.assertTrue(self.dialog.solve_button.isEnabled())
        with patch.object(self.dialog, '_run_astap', side_effect=self.solved_result), patch.object(QtWidgets.QMessageBox, 'information'), patch.object(QtWidgets.QMessageBox, 'critical') as failed:
            self.dialog._solve_current_image()
        failed.assert_not_called()
        self.assertEqual(len(self.siril.applied_headers), 1)

    def test_escape_and_close_request_cancellation_without_destroying_dialog(self):
        for action in (self.dialog.reject, self.dialog.close):
            self.dialog._set_busy(True)
            action()
            self.assertTrue(self.dialog._cancel_requested)
            self.assertTrue(self.dialog._busy)
            self.dialog._set_busy(False)

    def test_failed_start_cleans_up_process(self):
        with self.assertRaisesRegex(RuntimeError, 'Could not start ASTAP'):
            self.dialog._run_astap(self.root/'missing', self.root/'working.fit', 'missing', 30, False)
        self.assertIsNone(self.dialog._astap_process)

    def test_stale_sidecars_removed_before_failed_attempt(self):
        path = self.root/'working.fit'
        path.with_suffix('.wcs').write_text('old solution')
        path.with_suffix('.ini').write_text('PLTSOLVD=T')
        executable = self.fake_executable('raise SystemExit(1)\n')
        with self.assertRaisesRegex(RuntimeError, 'did not produce an .ini'):
            self.dialog._run_astap(executable, path, 'failure', 30, False)
        self.assertFalse(path.with_suffix('.wcs').exists())
        self.assertIsNone(self.dialog._astap_process)


if __name__ == '__main__':
    unittest.main()
