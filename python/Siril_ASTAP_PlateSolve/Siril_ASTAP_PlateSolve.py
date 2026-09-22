#!/usr/bin/env python3
"""
ASTAP plate-solving helper for Siril 1.4.

This script exports the current Siril image to a temporary FITS, asks ASTAP to
solve and update that FITS header, then imports the solved result back into the
current Siril image. A scaled mono proxy is used only for solving; original
pixels and non-astrometric metadata are preserved. FITS sources are overwritten
in place; TIFF sources are saved as a sibling solved FITS file.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import sirilpy as s
from sirilpy import NoImageError, ProcessingThreadBusyError, SirilConnectionError, SirilError

s.ensure_installed("PyQt6")

from PyQt6 import QtCore, QtWidgets


VERSION = "0.2.1"
ASTAP_TIMEOUT_SECONDS = 180
WINDOW_TITLE = f"ASTAP Plate Solve for Siril {VERSION}"
DEFAULT_RADIUS_DEG = 30
DEFAULT_BLIND_RADIUS_DEG = 180
FITS_SUFFIXES = {".fit", ".fits", ".fts"}
TIFF_SUFFIXES = {".tif", ".tiff"}
DEFAULT_ASTAP_NAMES = (
    "astap",
    "astap_cli",
    "astap.exe",
    "astap_cli.exe",
    "ASTAP",
    "ASTAP_CLI",
)


def parse_key_value_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split("//", 1)[0].strip().strip('"')
        values[key.strip().upper()] = value
    return values


def prepare_solver_pixels(pixels: np.ndarray) -> np.ndarray:
    """Same-size mono proxy with explicit ADU range; never modify source pixels."""
    if pixels.ndim == 3 and pixels.shape[0] in (1, 3):
        mono = np.mean(pixels, axis=0, dtype=np.float32)
    elif pixels.ndim == 2:
        mono = np.array(pixels, dtype=np.float32, copy=True)
    else:
        raise RuntimeError("Expected a mono image or channel-first RGB image.")
    finite = np.isfinite(mono)
    if not finite.any():
        raise RuntimeError("The image contains no finite pixels.")
    low = float(np.min(mono, where=finite, initial=np.inf))
    high = float(np.max(mono, where=finite, initial=-np.inf))
    if high <= low:
        raise RuntimeError("The image is constant; no stars can be detected.")
    mono[~finite] = low
    # ASTAP can mistake normalized floats with peaks above 1 for ADU data.
    # A linear conversion preserves star profiles, unlike stretching/clipping.
    mono -= low
    mono *= 65535.0 / (high - low)
    return np.ascontiguousarray(mono)


_ASTROMETRY_KEY = re.compile(
    r"(?:WCSAXES|WCSNAME|CTYPE[12]|CUNIT[12]|CRPIX[12]|CRVAL[12]|"
    r"CDELT[12]|CROTA[12]|(?:CD|PC)[12]_[12]|PV[12]_\d+|"
    r"(?:A|B|AP|BP)_(?:ORDER|DMAX|\d+_\d+)|"
    r"LONPOLE|LATPOLE|RADESYS|RADECSYS|EQUINOX|PLTSOLVD|RA|DEC)"
)


def header_cards(header: str) -> list[str]:
    # Siril's shared-memory header includes its C-string terminator. Keeping
    # END\0 would hide all subsequently appended WCS cards from Siril's parser.
    header = header.partition("\x00")[0]
    if "\n" in header:
        cards = header.splitlines()
    else:
        cards = [header[i:i + 80] for i in range(0, len(header), 80)]
    result = []
    for card in cards:
        if card.strip():
            result.append(card)
        if card[:8].strip() == "END":
            break
    return result


def merge_astrometry(original: str, solved: str) -> str:
    """Copy only sky coordinates, never mono geometry or proxy intensity metadata."""
    solution = [card for card in header_cards(solved)
                if _ASTROMETRY_KEY.fullmatch(card[:8].strip())]
    keys = {card[:8].strip() for card in solution}
    if not {"CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CD1_1", "CD2_2"} <= keys:
        raise RuntimeError("ASTAP returned an incomplete WCS header.")
    kept = [card for card in header_cards(original)
            if not _ASTROMETRY_KEY.fullmatch(card[:8].strip())
            and card[:8].strip() not in {"END", "CHECKSUM", "DATASUM"}]
    return "\n".join(kept + solution + ["END".ljust(80)])


class AstapPlateSolveDialog(QtWidgets.QDialog):
    def __init__(self, siril: s.SirilInterface) -> None:
        super().__init__()
        self.siril = siril
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
        path_label = QtWidgets.QLabel("ASTAP executable")
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Path to astap / astap_cli executable")
        browse_button = QtWidgets.QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_astap_executable)
        path_row.addWidget(path_label)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_button)
        layout.addLayout(path_row)

        self.solve_button = QtWidgets.QPushButton("Solve")
        self.solve_button.clicked.connect(self._solve_current_image)
        layout.addWidget(self.solve_button)

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

    def _save_config(self) -> None:
        config = {
            "astap_path": self.path_edit.text().strip(),
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
        self.solve_button.setEnabled(not busy)
        self.path_edit.setEnabled(not busy)
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QApplication.processEvents()

    def _resolve_astap_executable(self, raw_path: str) -> Path:
        cleaned = raw_path.strip().strip('"')
        if not cleaned:
            raise FileNotFoundError("ASTAP executable path is empty.")

        candidate = Path(os.path.expanduser(cleaned))
        search_paths: list[Path] = []

        if candidate.exists() and candidate.is_file():
            search_paths.append(candidate)

        if candidate.exists() and candidate.is_dir():
            for name in DEFAULT_ASTAP_NAMES:
                search_paths.append(candidate / name)
                search_paths.append(candidate / "Contents" / "MacOS" / name)

        if candidate.suffix.lower() == ".app":
            for name in DEFAULT_ASTAP_NAMES:
                search_paths.append(candidate / "Contents" / "MacOS" / name)

        for path in search_paths:
            if path.exists() and path.is_file():
                return path

        raise FileNotFoundError(
            "Could not find an ASTAP executable at the specified location."
        )

    def _run_astap(
        self,
        astap_path: Path,
        input_path: Path,
        strategy_name: str,
        radius_deg: int,
        use_auto_fov: bool,
    ) -> tuple[dict[str, str], str]:
        for sidecar in (".ini", ".wcs", ".log"):
            sidecar_path = input_path.with_suffix(sidecar)
            if sidecar_path.exists():
                sidecar_path.unlink()

        command = [
            str(astap_path),
            "-f",
            str(input_path),
            "-r",
            str(radius_deg),
            "-z",
            "0",
            "-sip",
            "-wcs",
            "-progress",
        ]
        if use_auto_fov:
            command.extend(["-fov", "0"])

        self._append_log(f"Strategy: {strategy_name}")
        self._append_log(" ".join(command))

        started = time.monotonic()
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.start(command[0], command[1:])
        if not process.waitForStarted(5000):
            raise RuntimeError(f"Could not start ASTAP: {process.errorString()}")
        output_lines: list[str] = []
        pending = b""

        def drain_output(final: bool = False) -> None:
            nonlocal pending
            pending += bytes(process.readAllStandardOutput())
            lines = pending.split(b"\n")
            pending = lines.pop()
            if final and pending:
                lines.append(pending)
                pending = b""
            for line in lines:
                clean_line = line.decode("utf-8", errors="replace").rstrip()
                output_lines.append(clean_line)
                self._append_log(clean_line)

        try:
            while process.state() != QtCore.QProcess.ProcessState.NotRunning:
                process.waitForFinished(50)
                drain_output()
                QtWidgets.QApplication.processEvents()
                if time.monotonic() - started > ASTAP_TIMEOUT_SECONDS:
                    raise RuntimeError(f"ASTAP timed out after {ASTAP_TIMEOUT_SECONDS} seconds.")
            drain_output(final=True)
            returncode = process.exitCode()
            if process.exitStatus() == QtCore.QProcess.ExitStatus.CrashExit:
                raise RuntimeError("ASTAP crashed while solving the image.")
        finally:
            if process.state() != QtCore.QProcess.ProcessState.NotRunning:
                process.kill()
                process.waitForFinished(5000)
            process.deleteLater()
        self._append_log(f"ASTAP elapsed time: {time.monotonic() - started:.1f} s")

        ini_path = input_path.with_suffix(".ini")
        if not ini_path.exists():
            raise RuntimeError("ASTAP did not produce an .ini result file.")

        ini_values = parse_key_value_text(ini_path.read_text(encoding="utf-8", errors="replace"))

        if returncode not in (0, 1, 2, 16, 32, 33):
            raise RuntimeError(f"ASTAP exited with unexpected code {returncode}.")

        return ini_values, "\n".join(output_lines)

    def _solve_current_image(self) -> None:
        self._refresh_loaded_image_label()
        self.log_box.clear()

        try:
            self._save_config()
            astap_path = self._resolve_astap_executable(self.path_edit.text())
        except (FileNotFoundError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "ASTAP path error", str(exc))
            return

        try:
            if not self.siril.is_image_loaded():
                raise NoImageError("No single image is currently loaded in Siril.")
        except SirilError as exc:
            QtWidgets.QMessageBox.critical(self, "Siril error", str(exc))
            return

        self._set_busy(True)

        try:
            current_name = self.siril.get_image_filename()
            if not current_name:
                raise RuntimeError("The current image must be associated with a FITS or TIFF file.")

            original_path = Path(current_name)
            source_suffix = original_path.suffix.lower()
            if source_suffix in FITS_SUFFIXES:
                output_path = original_path
            elif source_suffix in TIFF_SUFFIXES:
                output_path = original_path.with_name(f"{original_path.stem}_astap_solved.fit")
                self._append_log(f"TIFF source detected. Solved result will be saved as FITS: {output_path}")
            else:
                raise RuntimeError("This script currently supports FITS and TIFF sources only.")

            pixeldata = self.siril.get_image_pixeldata()
            if pixeldata is None:
                raise RuntimeError("Could not read pixel data from the current Siril image.")
            header = self.siril.get_image_fits_header()
            if not isinstance(header, str):
                raise RuntimeError("Could not read the FITS header from the current Siril image.")

            original_pixels = np.ascontiguousarray(pixeldata)
            working_pixels = prepare_solver_pixels(original_pixels)
            self._append_log("Solving a same-size monochrome proxy scaled to 0–65535; original pixels are preserved.")

            with tempfile.TemporaryDirectory(prefix="siril_astap_") as tmpdir_name:
                working_path = Path(tmpdir_name) / "astap_work.fit"

                def solve_with_strategies(strategy_pixels: np.ndarray) -> tuple[dict[str, str], str]:
                    if not self.siril.save_image_file(strategy_pixels, header, str(working_path)):
                        raise RuntimeError("Could not write the temporary solver image.")
                    self._append_log(f"Working FITS written to {working_path}")

                    attempts = (
                        ("Header-guided solve", DEFAULT_RADIUS_DEG, False),
                        ("Wide search with header FOV", DEFAULT_BLIND_RADIUS_DEG, False),
                        ("Blind auto-FOV solve", DEFAULT_BLIND_RADIUS_DEG, True),
                    )
                    latest_ini: dict[str, str] | None = None

                    for strategy_name, radius_deg, use_auto_fov in attempts:
                        latest_ini, _ = self._run_astap(
                            astap_path,
                            working_path,
                            strategy_name,
                            radius_deg,
                            use_auto_fov,
                        )
                        if latest_ini.get("PLTSOLVD", "").upper() == "T":
                            wcs_path = working_path.with_suffix(".wcs")
                            if not wcs_path.exists():
                                raise RuntimeError("ASTAP did not produce a WCS header.")
                            solved_header = wcs_path.read_text(encoding="ascii")
                            return latest_ini, merge_astrometry(header, solved_header)

                        error_message = latest_ini.get("ERROR", "ASTAP did not return a solution.")
                        warning_message = latest_ini.get("WARNING", "")
                        self._append_log(f"{strategy_name} failed: {error_message}")
                        if warning_message:
                            self._append_log(f"{strategy_name} warning: {warning_message}")
                        if "not enough stars" in error_message.lower() or "no stars" in error_message.lower():
                            self._append_log("Stopping: a wider sky search cannot fix insufficient star detection.")
                            break

                    assert latest_ini is not None
                    error_message = latest_ini.get("ERROR", "ASTAP did not return a solution.")
                    warning_message = latest_ini.get("WARNING", "")
                    full_message = error_message if not warning_message else f"{error_message}\nWarning: {warning_message}"
                    raise RuntimeError(full_message)

                ini_values, solved_header = solve_with_strategies(working_pixels)

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

            if source_suffix in TIFF_SUFFIXES:
                completion_message = (
                    f"ASTAP solved the TIFF image via temporary FITS and saved the result as:\n{output_path}"
                )
            else:
                completion_message = f"ASTAP solved and updated:\n{output_path}"

            QtWidgets.QMessageBox.information(
                self,
                "Plate solve complete",
                completion_message,
            )
        except ProcessingThreadBusyError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Siril busy",
                f"Siril is busy with another image operation: {exc}",
            )
        except (NoImageError, SirilError, OSError, RuntimeError) as exc:
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
