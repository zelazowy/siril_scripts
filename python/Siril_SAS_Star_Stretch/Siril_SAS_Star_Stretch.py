# SPDX-License-Identifier: GPL-3.0-only
# Siril adaptation Copyright (C) 2026 zelazowy
# Star Stretch algorithm reference: Franklin Marek and SAS Pro contributors.
# Adapted for Siril on 2026-09-27; see NOTICE.md for provenance and changes.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 3 as published
# by the Free Software Foundation.
# This program is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the accompanying LICENSE, or https://www.gnu.org/licenses/gpl-3.0.html.

"""
SAS-style Star Stretch for Siril 1.4.x.

Implements the mathematical transform used by Seti Astro Suite Pro:
f = 3 ** amount; output = f*x / (1 + (f-1)*x), independently per channel,
followed by mean-based color boost and optional average-neutral SCNR.
Algorithm reference: Franklin Marek / setiastro/setiastrosuitepro,
src/setiastro/saspro/star_stretch.py and legacy/numba_utils.py.
https://github.com/setiastro/setiastrosuitepro
This standalone Siril implementation does not launch SAS Pro.
Run on a stars-only image. Apply updates Siril's image with an undo state;
it does not save over the source file. Preview uses a downscaled image.
"""
import sys
import traceback

import sirilpy as s
from sirilpy import LogColor

s.ensure_installed("numpy", "opencv-python", "PyQt6")

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

VERSION = "1.0.0"

DARK_STYLESHEET = """
QWidget {
    background-color: #1d1d1d;
    color: #e0e0e0;
    font-size: 10pt;
}
QLabel {
    color: #e0e0e0;
}
QPushButton {
    background-color: #2f2f2f;
    color: #e0e0e0;
    border: 1px solid #4c4c4c;
    border-radius: 4px;
    padding: 6px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #393939;
}
QPushButton#ApplyButton {
    background-color: #245b9b;
    border-color: #346fb4;
}
QPushButton#ApplyButton:hover {
    background-color: #2c67aa;
}
QPushButton:disabled {
    color: #777777;
    background-color: #282828;
}
QSlider::groove:horizontal {
    background: #404040;
    height: 5px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #b0b0b0;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 1px solid #5c5c5c;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #5c5c5c;
    background: #2a2a2a;
}
QCheckBox::indicator:checked {
    background: #2d74ca;
    border: 1px solid #66a6ff;
}
QGraphicsView {
    background-color: #050505;
    border: 1px solid #3f3f3f;
}
"""


def _nofocus(widget):
    try:
        widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    except Exception:
        pass


class ResettableSlider(QSlider):
    def __init__(self, orientation, default_value, parent=None):
        super().__init__(orientation, parent)
        self.default_value = int(default_value)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setValue(self.default_value)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class StretchCore:
    @staticmethod
    def normalize_input(data):
        image = np.asarray(data)
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif not np.issubdtype(image.dtype, np.floating):
            raise ValueError(f"Unsupported Siril pixel type: {image.dtype}")
        if not np.isfinite(image).all():
            raise ValueError("Image contains NaN or infinite pixels.")
        return np.clip(image, 0, 1).astype(np.float32)

    @staticmethod
    def ensure_hwc(image):
        # Siril's API uses channels first, including narrow RGB images.
        if image.ndim == 2:
            return np.repeat(image[..., None], 3, axis=2)
        if image.ndim == 3 and image.shape[0] == 1:
            return np.repeat(image[0, ..., None], 3, axis=2)
        if image.ndim == 3 and image.shape[0] == 3:
            return np.moveaxis(image, 0, -1)
        raise ValueError(f"Unsupported Siril image shape: {image.shape}")

    @staticmethod
    def apply_sas_stretch(image, amount, color_boost=1.0, scnr=False):
        factor = 3.0 ** float(amount)
        out = np.clip(factor * image / (1.0 + (factor - 1.0) * image), 0, 1)
        if abs(color_boost - 1.0) > 1e-6:
            mean = out.mean(axis=2, keepdims=True)
            out = np.clip(mean + (out - mean) * color_boost, 0, 1)
        if scnr:
            out[..., 1] = np.minimum(out[..., 1], (out[..., 0] + out[..., 2]) * 0.5)
        return out.astype(np.float32, copy=False)


class StarStretchDialog(QMainWindow):
    PREVIEW_MAX_WIDTH = 2200
    PREVIEW_MAX_HEIGHT = 1400
    DEFAULT_STRETCH_SLIDER = 500

    def __init__(self, siril):
        super().__init__()
        self.siril = siril

        self.img_full = None
        self.img_proxy = None
        self.processed_proxy = None
        self.is_mono_source = False
        self.proxy_scale = 1.0

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(120)
        self.preview_timer.timeout.connect(self.run_preview)

        self.setWindowTitle("SAS Star Stretch")
        self.resize(1380, 900)
        self.setStyleSheet(DARK_STYLESHEET)

        self._build_ui()
        self.cache_input()

    @property
    def stretch_amount(self):
        return self.slider_stretch.value() / 100.0

    @property
    def saturation(self):
        return self.slider_sat.value() / 100.0

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        outer = QHBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(10)

        title = QLabel("SAS Star Stretch")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        side.addWidget(title)

        desc = QLabel("SAS Pro’s star stretch transform. Use on a stars-only image. "
                      "Preview is downscaled; Apply processes the full image.")
        desc.setWordWrap(True)
        side.addWidget(desc)

        self.lbl_stretch = QLabel()
        side.addWidget(self.lbl_stretch)

        self.slider_stretch = ResettableSlider(
            Qt.Orientation.Horizontal,
            self.DEFAULT_STRETCH_SLIDER,
        )
        self.slider_stretch.setRange(0, 800)
        self.slider_stretch.setValue(self.DEFAULT_STRETCH_SLIDER)
        self.slider_stretch.valueChanged.connect(self.on_controls_changed)
        _nofocus(self.slider_stretch)
        side.addWidget(self.slider_stretch)

        self.lbl_sat = QLabel()
        side.addWidget(self.lbl_sat)

        self.slider_sat = ResettableSlider(Qt.Orientation.Horizontal, 100)
        self.slider_sat.setRange(0, 200)
        self.slider_sat.setValue(100)
        self.slider_sat.valueChanged.connect(self.on_controls_changed)
        _nofocus(self.slider_sat)
        side.addWidget(self.slider_sat)

        self.chk_scnr = QCheckBox("Remove Green (SCNR)")
        self.chk_scnr.toggled.connect(self.on_controls_changed)
        _nofocus(self.chk_scnr)
        side.addWidget(self.chk_scnr)

        self.chk_auto = QCheckBox("Auto Preview")
        self.chk_auto.setChecked(True)
        _nofocus(self.chk_auto)
        side.addWidget(self.chk_auto)

        row_a = QHBoxLayout()

        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self.run_preview)
        _nofocus(self.btn_preview)
        row_a.addWidget(self.btn_preview)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_controls)
        _nofocus(self.btn_reset)
        row_a.addWidget(self.btn_reset)

        side.addLayout(row_a)

        row_b = QHBoxLayout()

        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setObjectName("ApplyButton")
        self.btn_apply.clicked.connect(self.apply_process)
        _nofocus(self.btn_apply)
        row_b.addWidget(self.btn_apply)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self.fit_view)
        _nofocus(self.btn_fit)
        row_b.addWidget(self.btn_fit)

        side.addLayout(row_b)

        row_c = QHBoxLayout()

        self.btn_zoom_out = QPushButton("Zoom Out")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        _nofocus(self.btn_zoom_out)
        row_c.addWidget(self.btn_zoom_out)

        self.btn_zoom_in = QPushButton("Zoom In")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        _nofocus(self.btn_zoom_in)
        row_c.addWidget(self.btn_zoom_in)

        side.addLayout(row_c)
        self.btn_license = QPushButton("License / About")
        self.btn_license.clicked.connect(self.show_license)
        side.addWidget(self.btn_license)
        side.addStretch()
        outer.addWidget(sidebar)

        scene_host = QWidget()
        scene_layout = QVBoxLayout(scene_host)
        scene_layout.setContentsMargins(0, 0, 0, 0)
        scene_layout.setSpacing(0)

        self.scene = QGraphicsScene(self)
        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        scene_layout.addWidget(self.view)

        outer.addWidget(scene_host, 1)

        self.update_labels()

    def show_license(self):
        QMessageBox.about(
            self, "About SAS-style Star Stretch",
            "<b>Unofficial Siril adaptation</b><br>"
            "Siril adaptation © 2026 zelazowy.<br>"
            "Algorithm reference: Franklin Marek and SAS Pro contributors.<br><br>"
            "Free software under GNU GPL version 3. You may redistribute and "
            "modify it under that license. Provided WITHOUT ANY WARRANTY.<br><br>"
            '<a href="https://www.gnu.org/licenses/gpl-3.0.html">Read GNU GPL v3</a><br>'
            '<a href="https://github.com/setiastro/setiastrosuitepro">SAS Pro source</a>'
        )

    def _set_busy(self, busy):
        for widget in (
            self.slider_stretch,
            self.slider_sat,
            self.chk_scnr,
            self.chk_auto,
            self.btn_preview,
            self.btn_reset,
            self.btn_apply,
            self.btn_fit,
            self.btn_zoom_out,
            self.btn_zoom_in,
        ):
            widget.setEnabled(not busy)

    def update_mode_ui(self):
        self.slider_sat.setEnabled(not self.is_mono_source)
        self.chk_scnr.setEnabled(not self.is_mono_source)

    def update_labels(self):
        self.lbl_stretch.setText(f"Stretch Amount: {self.stretch_amount:.2f}")
        self.lbl_sat.setText(f"Color Boost: {self.saturation:.2f}")
        self.update_mode_ui()

    def on_controls_changed(self):
        self.update_labels()
        if self.chk_auto.isChecked():
            self.preview_timer.start()

    def reset_controls(self):
        self.slider_stretch.setValue(self.DEFAULT_STRETCH_SLIDER)
        self.slider_sat.setValue(100)
        self.chk_scnr.setChecked(False)
        self.run_preview()

    def apply_selected_mode(self, image, saturation, use_scnr):
        return StretchCore.apply_sas_stretch(image, self.stretch_amount, saturation, use_scnr)

    def cache_input(self):
        try:
            if not self.siril.connected:
                self.siril.connect()

            with self.siril.image_lock():
                img = self.siril.get_image_pixeldata()

            if img is None:
                raise RuntimeError("No active image is open in Siril.")

            self.is_mono_source = (
                img.ndim == 2
                or (img.ndim == 3 and img.shape[0] == 1)
            )

            self.source_pixels = img.copy()
            self.source_shape = img.shape
            normalized = StretchCore.normalize_input(img)
            self.img_full = StretchCore.ensure_hwc(normalized)
            self.img_proxy = self.build_preview_proxy(self.img_full)
            self.processed_proxy = self.img_proxy.copy()

            if self.is_mono_source:
                self.slider_sat.setEnabled(False)
                self.chk_scnr.setEnabled(False)

            self.update_mode_ui()
            self.update_view(self.img_proxy)
            QTimer.singleShot(50, self.fit_view)
            QTimer.singleShot(150, self.run_preview)

            self.siril.log(f"SAS Star Stretch v{VERSION} loaded.", LogColor.GREEN)

        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(exc))
            self.close()

    def get_preview_target_size(self):
        screen = None
        try:
            handle = self.windowHandle()
            if handle is not None:
                screen = handle.screen()
        except Exception:
            screen = None

        if screen is None:
            screen = QApplication.primaryScreen()

        if screen is None:
            return 1920, 1080

        geom = screen.availableGeometry()
        dpr = 1.0
        try:
            dpr = max(1.0, float(screen.devicePixelRatio()))
        except Exception:
            pass

        target_w = max(1, min(int(geom.width() * dpr), self.PREVIEW_MAX_WIDTH))
        target_h = max(1, min(int(geom.height() * dpr), self.PREVIEW_MAX_HEIGHT))
        return target_w, target_h

    def build_preview_proxy(self, img):
        height, width = img.shape[:2]
        target_w, target_h = self.get_preview_target_size()
        scale = min(1.0, target_w / max(1, width), target_h / max(1, height))
        self.proxy_scale = scale

        if scale >= 0.999:
            return img.copy()

        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def run_preview(self):
        if self.img_proxy is None:
            return

        try:
            use_scnr = (not self.is_mono_source) and self.chk_scnr.isChecked()
            saturation = self.saturation if not self.is_mono_source else 1.0
            stretched = self.apply_selected_mode(
                self.img_proxy,
                saturation,
                use_scnr,
            )

            self.processed_proxy = stretched
            self.update_view(self.processed_proxy)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Preview Error", str(exc))

    def update_view(self, data):
        disp = np.clip(data * 255.0, 0, 255).astype(np.uint8)
        disp = np.flipud(disp)
        disp = np.ascontiguousarray(disp)

        qimg = QImage(
            disp.data,
            disp.shape[1],
            disp.shape[0],
            disp.shape[1] * 3,
            QImage.Format.Format_RGB888,
        )
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, disp.shape[1], disp.shape[0])

    def zoom_in(self):
        self.view.scale(1.25, 1.25)

    def zoom_out(self):
        self.view.scale(0.8, 0.8)

    def fit_view(self):
        if self.pix_item.pixmap().isNull():
            return
        self.view.resetTransform()
        self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def apply_process(self):
        if self.img_full is None:
            return

        self._set_busy(True)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            saturation = self.saturation if not self.is_mono_source else 1.0
            use_scnr = (not self.is_mono_source) and self.chk_scnr.isChecked()
            finished = self.apply_selected_mode(
                self.img_full,
                saturation,
                use_scnr,
            )

            if self.is_mono_source:
                out = finished[..., 0].copy()
                if self.source_shape != out.shape:
                    out = out.reshape(self.source_shape)
            else:
                out = np.transpose(finished, (2, 0, 1)).astype(np.float32, copy=False)

            with self.siril.image_lock():
                current = self.siril.get_image_pixeldata()
                if current is None or current.shape != self.source_shape or not np.array_equal(current, self.source_pixels):
                    raise RuntimeError("The active image changed. Close this dialog and run the script again.")
                self.siril.undo_save_state("SAS Star Stretch")
                self.siril.set_image_pixeldata(out)

            self.siril.log("SAS Star Stretch applied (image not saved to disk).", LogColor.GREEN)
            self.close()

        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Apply Error", str(exc))
            self._set_busy(False)
        finally:
            QApplication.restoreOverrideCursor()


def main():
    app = QApplication(sys.argv)
    siril = s.SirilInterface()

    try:
        siril.connect()
    except Exception:
        pass

    gui = StarStretchDialog(siril)
    gui.show()
    app.exec()


if __name__ == "__main__":
    main()
