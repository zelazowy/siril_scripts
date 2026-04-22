#!/usr/bin/env python3
"""
ASTAP plate-solving helper for Siril 1.4.

This script exports the current Siril image to a temporary FITS, asks ASTAP to
solve and update that FITS header, then imports the solved result back into the
current Siril image and overwrites the source FITS file.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import sirilpy as s
from sirilpy import NoImageError, ProcessingThreadBusyError, SirilConnectionError, SirilError

s.ensure_installed("PyQt6")

from PyQt6 import QtCore, QtWidgets


VERSION = "0.1.0"
WINDOW_TITLE = f"ASTAP Plate Solve for Siril {VERSION}"
DEFAULT_RADIUS_DEG = 30
DEFAULT_BLIND_RADIUS_DEG = 180
FITS_SUFFIXES = {".fit", ".fits", ".fts"}
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
            "-update",
            "-wcs",
            "-progress",
        ]
        if use_auto_fov:
            command.extend(["-fov", "0"])

        self._append_log(f"Strategy: {strategy_name}")
        self._append_log(" ".join(command))

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        output_lines: list[str] = []
        assert process.stdout is not None

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            events = selector.select(timeout=0.2)
            if events:
                line = process.stdout.readline()
                if line:
                    clean_line = line.rstrip()
                    output_lines.append(clean_line)
                    self._append_log(clean_line)
            if process.poll() is not None:
                break
            QtWidgets.QApplication.processEvents()
        selector.unregister(process.stdout)
        selector.close()

        trailing_output = process.stdout.read()
        if trailing_output:
            for line in trailing_output.splitlines():
                output_lines.append(line)
                self._append_log(line)

        ini_path = input_path.with_suffix(".ini")
        if not ini_path.exists():
            raise RuntimeError("ASTAP did not produce an .ini result file.")

        ini_values = parse_key_value_text(ini_path.read_text(encoding="utf-8", errors="replace"))

        if process.returncode not in (0, 1, 2, 16, 32, 33):
            raise RuntimeError(f"ASTAP exited with unexpected code {process.returncode}.")

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
                raise RuntimeError("The current image must be associated with a FITS file.")

            original_path = Path(current_name)
            if original_path.suffix.lower() not in FITS_SUFFIXES:
                raise RuntimeError("This script currently supports overwriting FITS files only.")

            pixeldata = self.siril.get_image_pixeldata()
            if pixeldata is None:
                raise RuntimeError("Could not read pixel data from the current Siril image.")
            header = self.siril.get_image_fits_header()
            if not isinstance(header, str):
                raise RuntimeError("Could not read the FITS header from the current Siril image.")

            original_pixels = np.ascontiguousarray(pixeldata)
            working_pixels = original_pixels

            with tempfile.TemporaryDirectory(prefix="siril_astap_") as tmpdir_name:
                working_path = Path(tmpdir_name) / "astap_work.fit"

                def solve_with_strategies(strategy_pixels: np.ndarray) -> tuple[dict[str, str], str]:
                    self.siril.save_image_file(strategy_pixels, header, str(working_path))
                    self._append_log(f"Working FITS written to {working_path}")

                    attempts = (
                        ("Header-guided solve", DEFAULT_RADIUS_DEG, False),
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
                            solved_fit = self.siril.load_image_from_file(str(working_path), with_pixels=True)
                            if solved_fit is None or solved_fit.header is None:
                                raise RuntimeError("Could not read the solved FITS produced by ASTAP.")
                            return latest_ini, solved_fit.header

                        error_message = latest_ini.get("ERROR", "ASTAP did not return a solution.")
                        warning_message = latest_ini.get("WARNING", "")
                        self._append_log(f"{strategy_name} failed: {error_message}")
                        if warning_message:
                            self._append_log(f"{strategy_name} warning: {warning_message}")

                    assert latest_ini is not None
                    error_message = latest_ini.get("ERROR", "ASTAP did not return a solution.")
                    warning_message = latest_ini.get("WARNING", "")
                    full_message = error_message if not warning_message else f"{error_message}\nWarning: {warning_message}"
                    raise RuntimeError(full_message)

                ini_values, _ = solve_with_strategies(working_pixels)

                solved_fit = self.siril.load_image_from_file(str(working_path), with_pixels=True)
                if solved_fit is None or solved_fit.data is None or solved_fit.header is None:
                    raise RuntimeError("Could not load the final solved FITS from ASTAP.")

                with self.siril.image_lock():
                    self.siril.undo_save_state("ASTAP plate solve")
                    self.siril.set_image_pixeldata(solved_fit.data)
                    self.siril.set_image_metadata_from_header_string(solved_fit.header)
                    self.siril.set_image_filename(str(original_path))

                self.siril.save_image_file(solved_fit.data, solved_fit.header, str(original_path))
                self._append_log(f"Original FITS updated: {original_path}")

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

            QtWidgets.QMessageBox.information(
                self,
                "Plate solve complete",
                f"ASTAP solved and updated:\n{original_path}",
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
