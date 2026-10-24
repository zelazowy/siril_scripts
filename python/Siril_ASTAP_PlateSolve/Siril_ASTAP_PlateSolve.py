#!/usr/bin/env python3
"""
ASTAP plate-solving helper for Siril 1.4.

This script exports the current Siril image to a temporary FITS, asks ASTAP to
solve and update that FITS header, then imports the solved result back into the
current Siril image. Scaled mono and compact-star proxies are used only for solving; original
pixels and non-astrometric metadata are preserved. FITS sources are overwritten
in place; TIFF and JPEG sources are saved as a sibling solved FITS file.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sirilpy as s
from sirilpy import NoImageError, ProcessingThreadBusyError, SirilConnectionError, SirilError

s.ensure_installed("PyQt6")

from PyQt6 import QtCore, QtWidgets
try:
    from astro_platesolve import SolveOptions, solve_image, validate_backend, resolve_options
    from astro_platesolve.astap import normalize_pixels, compact_star_proxy
except ImportError as exc:
    raise RuntimeError("Install the shared astro-platesolve library in Siril's Python environment; see README.") from exc



VERSION = "0.6.0"
ASTAP_TIMEOUT_SECONDS = 180
WINDOW_TITLE = f"Plate Solve for Siril {VERSION}"
DEFAULT_RADIUS_DEG = 30
DEFAULT_BLIND_RADIUS_DEG = 180
FITS_SUFFIXES = {".fit", ".fits", ".fts"}
TIFF_SUFFIXES = {".tif", ".tiff"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}
class SolveCancelled(InterruptedError):
    """A user cancellation, handled without showing a failure dialog."""


def prepare_solver_pixels(pixels):
    return normalize_pixels(pixels)


def prepare_compact_star_pixels(mono, check_cancelled=lambda: None):
    class Cancel:
        def is_set(self): check_cancelled(); return False
    result = compact_star_proxy(mono, Cancel())
    return normalize_pixels(result) if result is not None else None


from astro_platesolve.metadata import header_cards, prepare_solver_header, ASTROMETRY_KEY as _ASTROMETRY_KEY


def prepare_jpeg_solver_header(
    path: Path, current_header: str, width: int, height: int,
) -> tuple[str, str]:
    """Use matching EXIF FITS cards as search hints, never as a final solution."""
    s.ensure_installed("Pillow")
    s.ensure_installed("astropy")
    from PIL import Image
    from astropy.io import fits
    from astropy.io.fits.verify import VerifyError
    from astropy.wcs import WCS

    prefix = "JPEG EXIF: "
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            description = exif.get(270)  # ImageDescription, used by Siril exports.
            if description is None:
                return current_header, prefix + "no embedded FITS header; solving from the loaded pixels."
            if exif.get(274, 1) != 1:
                return current_header, prefix + "rotated/mirrored orientation; ignoring embedded hints."
            if image.size != (width, height):
                return current_header, prefix + "loaded dimensions differ from the file; ignoring embedded hints."
        if isinstance(description, bytes):
            description = description.decode("ascii", errors="replace")
        if not isinstance(description, str):
            return current_header, prefix + "no readable FITS description; solving from the loaded pixels."
        description = description.partition("\x00")[0].strip().replace("\r\n", "\n").replace("\r", "\n")
        cards = header_cards(description)
        if not cards or cards[0][:8].strip() != "SIMPLE" or cards[-1][:8].strip() != "END":
            return current_header, prefix + "description is not a complete FITS header; ignoring it."
        embedded_text = prepare_solver_header(description)
        embedded = fits.Header.fromstring(embedded_text, sep="\n")
        for card in embedded.cards:
            card.verify("exception")
        if (embedded.get("NAXIS1"), embedded.get("NAXIS2")) != (width, height):
            return current_header, prefix + "embedded dimensions are stale; ignoring embedded hints."
        # As in SPACER, reject an invalid coordinate grid. ASTAP still verifies
        # the photographed stars, since dimension checks cannot detect all edits.
        embedded_has_wcs = "CTYPE1" in embedded or "CTYPE2" in embedded
        if embedded_has_wcs:
            wcs = WCS(embedded, naxis=2)
            scale = wcs.pixel_scale_matrix
            if (not wcs.has_celestial or not np.isfinite(scale).all()
                    or abs(np.linalg.det(scale)) < 1e-15
                    or not np.isfinite(wcs.all_pix2world(
                        [[0, 0], [width - 1, height - 1]], 0,
                    )).all()):
                return current_header, prefix + "invalid embedded WCS; ignoring embedded hints."
        hint_keys = {"FOCALLEN", "XPIXSZ", "YPIXSZ", "XBINNING", "YBINNING", "OBJCTRA", "OBJCTDEC"}
        # Treat WCS as one unit: mixing half a current solution with half an
        # embedded one can create contradictory coordinate matrices.
        current_cards = header_cards(current_header)
        current_has_wcs = False
        try:
            current = WCS(fits.Header.fromstring(prepare_solver_header(current_header), sep="\n"), naxis=2)
            current_has_wcs = (current.has_celestial and np.isfinite(current.pixel_scale_matrix).all()
                               and abs(np.linalg.det(current.pixel_scale_matrix)) >= 1e-15
                               and np.isfinite(current.all_pix2world([[0, 0], [width - 1, height - 1]], 0)).all())
        except (ValueError, TypeError, VerifyError):
            pass
        if embedded_has_wcs and not current_has_wcs:
            current_cards = [card for card in current_cards
                             if not _ASTROMETRY_KEY.fullmatch(card[:8].strip())]
        current_keys = {card[:8].strip() for card in current_cards}
        hints = [card for card in header_cards(embedded_text)
                 if (card[:8].strip() in hint_keys or
                     (not current_has_wcs and _ASTROMETRY_KEY.fullmatch(card[:8].strip())))
                 and card[:8].strip() not in current_keys]
        if not hints:
            return current_header, prefix + "no additional search hints; using Siril's current metadata."
        result = "\n".join([card for card in current_cards if card[:8].strip() != "END"]
                           + hints + ["END".ljust(80)])
        return result, prefix + "using embedded astrometry/optics to guide a fresh ASTAP solve."
    except (OSError, ValueError, TypeError, KeyError, VerifyError) as exc:
        return current_header, prefix + f"could not use embedded metadata ({exc}); solving from the loaded pixels."


def image_is_mirrored(header: str) -> bool:
    """Use the same celestial CD-matrix parity convention as Siril."""
    values = {}
    for card in header_cards(header):
        if card[8:10] == "= ":
            values[card[:8].strip()] = card[10:].split("/", 1)[0].strip()

    def number(key: str, default=None) -> float:
        value = values.get(key, default)
        if value is None:
            raise RuntimeError(f"Missing WCS keyword: {key}")
        return float(str(value).replace("D", "E"))

    if any(key in values for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        matrix = np.array([[number("CD1_1", 0), number("CD1_2", 0)],
                           [number("CD2_1", 0), number("CD2_2", 0)]])
    else:
        matrix = np.array([[number("PC1_1", 1), number("PC1_2", 0)],
                           [number("PC2_1", 0), number("PC2_2", 1)]])
        matrix *= np.array([number("CDELT1"), number("CDELT2")])[:, None]
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or determinant == 0:
        raise RuntimeError("Invalid WCS matrix; cannot determine image mirroring.")
    return determinant > 0


def merge_astrometry(original, solved):
    from astropy.io import fits
    from astro_platesolve.metadata import merge_headers
    source=fits.Header.fromstring("\n".join(header_cards(original)),sep="\n")
    solution=fits.Header.fromstring("\n".join(header_cards(solved)),sep="\n")
    return merge_headers(source,solution).tostring(sep="\n",endcard=True,padding=False)


class AstapPlateSolveDialog(QtWidgets.QDialog):
    def __init__(self, siril: s.SirilInterface) -> None:
        super().__init__()
        self.siril = siril
        self._busy = False
        self._cancel_requested = False
        self._committing = False
        self._astap_process = None
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumWidth(760)

        self.config_path = self._build_config_path()
        self._build_ui()
        self._load_config()
        self._refresh_loaded_image_label()

    def _build_config_path(self) -> Path:
        user_dir = Path(self.siril.get_siril_userdatadir())
        config_dir = user_dir / "scripts"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "astap_plate_solve_config.json"

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.loaded_image_label = QtWidgets.QLabel("Loaded image: <unknown>")
        self.loaded_image_label.setWordWrap(True)
        layout.addWidget(self.loaded_image_label)

        path_row = QtWidgets.QHBoxLayout()
        path_label = QtWidgets.QLabel("ASTAP fallback executable")
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Path to astap / astap_cli executable")
        self.browse_button = QtWidgets.QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_astap_executable)
        path_row.addWidget(path_label)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse_button)
        layout.addLayout(path_row)

        self.correct_mirror_check = QtWidgets.QCheckBox("Correct mirrored image if needed")
        self.correct_mirror_check.setChecked(True)
        self.correct_mirror_check.setToolTip(
            "Match Siril's plate-solving flip convention. Preserves pixel values "
            "and updates the coordinate solution; does not rotate north-up."
        )
        layout.addWidget(self.correct_mirror_check)

        self.solve_button = QtWidgets.QPushButton("Solve")
        self.solve_button.clicked.connect(self._solve_current_image)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_solve)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.solve_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Solver log")
        layout.addWidget(self.log_box, 1)

    def _browse_astap_executable(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ASTAP executable",
            str(Path.home()),
        )
        if path:
            self.path_edit.setText(path)

    def _refresh_loaded_image_label(self) -> None:
        try:
            if not self.siril.is_image_loaded():
                self.loaded_image_label.setText("Loaded image: none")
                return
            filename = self.siril.get_image_filename() or "<unsaved image>"
            self.loaded_image_label.setText(f"Loaded image: {filename}")
        except SirilError as exc:
            self.loaded_image_label.setText(f"Loaded image: unavailable ({exc})")

    def _load_config(self) -> None:
        if not self.config_path.exists():
            return
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.path_edit.setText(config.get("astap_path", ""))
        self.correct_mirror_check.setChecked(config.get("correct_mirror", True))

    def _save_config(self) -> None:
        config = {
            "astap_path": self.path_edit.text().strip(),
            "correct_mirror": self.correct_mirror_check.isChecked(),
        }
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def _append_log(self, message: str) -> None:
        self.log_box.appendPlainText(message)
        try:
            self.siril.log(message)
        except SirilError:
            pass
        QtWidgets.QApplication.processEvents()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._cancel_requested = False
        self._committing = False
        self.solve_button.setEnabled(not busy)
        self.cancel_button.setText("Cancel")
        self.cancel_button.setEnabled(busy)
        self.path_edit.setEnabled(not busy)
        self.browse_button.setEnabled(not busy)
        self.correct_mirror_check.setEnabled(not busy)
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QApplication.processEvents()

    def _cancel_solve(self) -> None:
        if not self._busy or self._committing or self._cancel_requested:
            return
        self._cancel_requested = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")
        self.log_box.appendPlainText("Cancellation requested; stopping ASTAP…")
        if self._astap_process is not None:
            self._astap_process.kill()

    def _check_cancelled(self) -> None:
        QtWidgets.QApplication.processEvents()
        if self._cancel_requested:
            raise SolveCancelled("Plate solving cancelled; no solution was applied or saved.")

    def reject(self) -> None:
        # Escape must not destroy a dialog while a nested solve is still running.
        if self._busy:
            self._cancel_solve()
        else:
            super().reject()

    def closeEvent(self, event) -> None:
        if self._busy:
            self._cancel_solve()
            event.ignore()
        else:
            super().closeEvent(event)

    def _solve_with_strategies(self, astap_path, working_path, mono, header, search_header=None):
        from astropy.io import fits
        search_header = header if search_header is None else search_header
        parsed = fits.Header.fromstring("\n".join(header_cards(search_header)), sep="\n")
        dialog = self
        class Cancel:
            def is_set(self):
                dialog._check_cancelled()
                return False
            def wait(self, seconds):
                dialog._check_cancelled()
                time.sleep(min(seconds, .05))
                dialog._check_cancelled()
                return False
        result = solve_image(mono, parsed, SolveOptions(
            executable=str(astap_path), timeout=ASTAP_TIMEOUT_SECONDS*3,
            log_path=str(working_path.with_suffix('.log'))), self._append_log, Cancel())
        self._append_log(f"Shared solver backend: {result.backend}; strategy: {result.strategy}")
        solved = result.header.tostring(sep="\n", endcard=True, padding=False)
        return {'PLTSOLVD':'T', 'BACKEND':result.backend}, merge_astrometry(header, solved)

    def _solve_current_image(self) -> None:
        if self._busy:
            return
        self._refresh_loaded_image_label()
        self.log_box.clear()

        try:
            self._save_config()
            configured = validate_backend(SolveOptions(executable=self.path_edit.text()))
            astap_path = Path(configured.executable or self.path_edit.text())
        except (FileNotFoundError, OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(self, "Solver configuration error", str(exc))
            return

        try:
            if not self.siril.is_image_loaded():
                raise NoImageError("No single image is currently loaded in Siril.")
        except SirilError as exc:
            QtWidgets.QMessageBox.critical(self, "Siril error", str(exc))
            return

        self._set_busy(True)

        try:
            self._check_cancelled()
            current_name = self.siril.get_image_filename()
            if not current_name:
                raise RuntimeError("The current image must be associated with a FITS, TIFF or JPEG file.")

            original_path = Path(current_name)
            source_suffix = original_path.suffix.lower()
            if source_suffix in FITS_SUFFIXES:
                output_path = original_path
            elif source_suffix in TIFF_SUFFIXES | JPEG_SUFFIXES:
                output_path = original_path.with_name(f"{original_path.stem}_astap_solved.fit")
                source_format = "JPEG" if source_suffix in JPEG_SUFFIXES else "TIFF"
                self._append_log(f"{source_format} source detected. Solved result will be saved as FITS: {output_path}")
            else:
                raise RuntimeError("This script currently supports FITS, TIFF and JPEG sources only.")

            pixeldata = self.siril.get_image_pixeldata()
            if pixeldata is None:
                raise RuntimeError("Could not read pixel data from the current Siril image.")
            header = self.siril.get_image_fits_header()
            if not isinstance(header, str):
                raise RuntimeError("Could not read the FITS header from the current Siril image.")

            original_pixels = np.ascontiguousarray(pixeldata)
            search_header = header
            if source_suffix in JPEG_SUFFIXES:
                self._check_cancelled()
                search_header, message = prepare_jpeg_solver_header(
                    original_path, header, original_pixels.shape[-1], original_pixels.shape[-2],
                )
                self._append_log(message)
            working_pixels = original_pixels
            self._check_cancelled()
            self._append_log("Using the shared plate-solving backend; original pixels are preserved.")

            with tempfile.TemporaryDirectory(prefix="siril_astap_") as tmpdir_name:
                working_path = Path(tmpdir_name) / "astap_work.fit"

                ini_values, solved_header = self._solve_with_strategies(
                    astap_path, working_path, working_pixels, header, search_header,
                )

                self._check_cancelled()
                # Applying WCS, mirroring and saving form one final operation.
                # A late cancellation must not leave a half-applied solution.
                self._committing = True
                self.cancel_button.setEnabled(False)
                with self.siril.image_lock():
                    self.siril.undo_save_state("ASTAP plate solve")
                    self.siril.set_image_metadata_from_header_string(solved_header)
                    self.siril.set_image_filename(str(output_path))

                # ASTAP success alone does not establish that Siril accepted WCS.
                try:
                    sky_position = self.siril.pix2radec(
                        original_pixels.shape[-1] / 2,
                        original_pixels.shape[-2] / 2,
                    )
                except ValueError as exc:
                    raise RuntimeError("ASTAP solved the image, but Siril did not accept the WCS.") from exc
                if sky_position is None or not np.isfinite(sky_position).all():
                    raise RuntimeError("Siril could not verify the imported sky coordinates.")
                self._append_log("Siril verified the imported plate solution.")

                if self.correct_mirror_check.isChecked():
                    if image_is_mirrored(solved_header):
                        self._append_log("Mirrored image detected; correcting with Siril's mirror command.")
                        # Run outside image_lock: Siril commands claim their own thread.
                        # Siril transforms both pixels and WCS, including SIP distortion.
                        self.siril.cmd("mirrorx")
                        original_pixels = self.siril.get_image_pixeldata()
                        solved_header = self.siril.get_image_fits_header()
                        if original_pixels is None or not isinstance(solved_header, str):
                            raise RuntimeError("Could not read the corrected image from Siril.")
                        if image_is_mirrored(solved_header):
                            raise RuntimeError("Siril did not correct the image mirroring.")
                        sky_position = self.siril.pix2radec(
                            original_pixels.shape[-1] / 2,
                            original_pixels.shape[-2] / 2,
                        )
                        if sky_position is None or not np.isfinite(sky_position).all():
                            raise RuntimeError("Could not verify WCS after mirror correction.")
                        self._append_log("Mirroring corrected; pixel values and sky coordinates preserved.")
                    else:
                        self._append_log("Image orientation is already unmirrored; no flip needed.")

                if not self.siril.save_image_file(original_pixels, solved_header, str(output_path)):
                    raise RuntimeError(f"Could not save the solved image to {output_path}")
                self._append_log(f"Solved FITS updated: {output_path}")

            warning_message = ini_values.get("WARNING", "")
            crval1 = ini_values.get("CRVAL1", "?")
            crval2 = ini_values.get("CRVAL2", "?")
            cdelt1 = ini_values.get("CDELT1", "")
            solved_message = f"Solved successfully. CRVAL1={crval1} CRVAL2={crval2}"
            if cdelt1:
                solved_message += f" CDELT1={cdelt1}"
            self._append_log(solved_message)
            if warning_message:
                self._append_log(f"ASTAP warning: {warning_message}")

            if source_suffix in TIFF_SUFFIXES | JPEG_SUFFIXES:
                completion_message = (
                    f"ASTAP solved the {source_format} image via temporary FITS and saved the result as:\n{output_path}"
                )
            else:
                completion_message = f"ASTAP solved and updated:\n{output_path}"

            QtWidgets.QMessageBox.information(
                self,
                "Plate solve complete",
                completion_message,
            )
        except SolveCancelled as exc:
            self._append_log(str(exc))
        except ProcessingThreadBusyError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Siril busy",
                f"Siril is busy with another image operation: {exc}",
            )
        except (NoImageError, SirilError, OSError, RuntimeError, ValueError) as exc:
            self._append_log(f"Error: {exc}")
            QtWidgets.QMessageBox.critical(self, "Plate solve failed", str(exc))
        finally:
            self._set_busy(False)
            self._refresh_loaded_image_label()


def main() -> int:
    siril = s.SirilInterface()
    try:
        siril.connect()
    except SirilConnectionError as exc:
        print(f"Could not connect to Siril: {exc}")
        return 1

    if siril.is_cli():
        siril.log("This script requires the Siril GUI because it provides a dialog.")
        return 1

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    dialog = AstapPlateSolveDialog(siril)
    return dialog.exec()


if __name__ == "__main__":
    raise SystemExit(main())
