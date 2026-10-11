# SPDX-License-Identifier: GPL-3.0-only
# Unofficial port of Seti Astro Suite Pro by Franklin Marek and contributors.
# Adapted for Siril on 2026-10-11; see NOTICE.md for sources and changes.
# This program is free software under GNU GPL version 3.
# Distributed WITHOUT ANY WARRANTY; see the accompanying LICENSE.
# Siril adaptation Copyright (C) 2026 zelazowy.
"""SAS-style star-based white balance for Siril 1.4; run before stretching."""
import sys
from pathlib import Path
import traceback
import sirilpy as s
s.ensure_installed('numpy', 'sep', 'PyQt6')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sas_white_balance_core import apply_star_based_white_balance, _to_float01, preview_rgb
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPen, QColor
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QCheckBox, QDoubleSpinBox, QGraphicsScene,
    QGraphicsView, QGraphicsPixmapItem, QMessageBox)

class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    def __init__(self, image, auto, threshold):
        super().__init__()
        self.image, self.auto, self.threshold = image, auto, threshold
    def run(self):
        try:
            import sas_white_balance_core as core
            used = []
            result = apply_star_based_white_balance(self.image, threshold=self.threshold,
                auto_sigma=self.auto, autostretch=False, on_sigma_resolved=used.append)
            self.done.emit((result[0], result[1], core.cached_star_sources.copy(), used[0]))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))

class WhiteBalanceWindow(QMainWindow):
    def __init__(self, siril):
        super().__init__()
        self.siril = siril
        self.worker = None
        self.result = None
        self.sources = None
        with siril.image_lock():
            pixels = siril.get_image_pixeldata()
            if pixels is None or pixels.ndim != 3 or pixels.shape[0] != 3:
                raise ValueError('Open a linear RGB image in Siril first.')
            self.original = pixels.copy()
        self.image = np.ascontiguousarray(_to_float01(self.original).transpose(1, 2, 0))
        self.preview_cache = {}
        self.setWindowTitle('SAS-style Star-Based White Balance — Siril')
        self.resize(1100, 800)
        host = QWidget(); self.setCentralWidget(host)
        layout = QVBoxLayout(host)
        layout.addWidget(QLabel('Run before stretching.'))
        row = QHBoxLayout(); layout.addLayout(row)
        self.auto = QCheckBox('Auto detection'); self.auto.setChecked(True)
        row.addWidget(self.auto)
        row.addWidget(QLabel('Threshold (σ):'))
        self.threshold = QDoubleSpinBox(); self.threshold.setRange(0.5, 100); self.threshold.setValue(5)
        self.threshold.setEnabled(False); row.addWidget(self.threshold)
        self.auto.toggled.connect(lambda value: self.threshold.setEnabled(not value))
        self.compute = QPushButton('Analyze / Preview'); row.addWidget(self.compute)
        self.compute.clicked.connect(self.analyze)
        self.before = QCheckBox('Show original'); row.addWidget(self.before)
        self.before.toggled.connect(self.render)
        self.overlay = QCheckBox('Show sampled stars'); self.overlay.setChecked(True)
        row.addWidget(self.overlay); self.overlay.toggled.connect(self.render)
        self.apply = QPushButton('Apply'); self.apply.setEnabled(False); row.addWidget(self.apply)
        self.apply.clicked.connect(self.apply_result)
        self.auto.toggled.connect(self.invalidate); self.threshold.valueChanged.connect(self.invalidate)
        self.status = QLabel('Click Analyze to calculate balance from full-resolution stars.')
        layout.addWidget(self.status)
        self.scene = QGraphicsScene(self); self.item = QGraphicsPixmapItem(); self.scene.addItem(self.item)
        self.view = QGraphicsView(self.scene); self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(self.view)
        self.render(); self.view.fitInView(self.item, Qt.AspectRatioMode.KeepAspectRatio)
    def invalidate(self, *_):
        self.preview_cache.pop('after', None)
        self.result = None; self.sources = None; self.apply.setEnabled(False); self.render()
    def analyze(self):
        self.compute.setEnabled(False); self.apply.setEnabled(False)
        self.auto.setEnabled(False); self.threshold.setEnabled(False)
        self.status.setText('Analyzing full-resolution image…')
        self.worker = Worker(self.image, self.auto.isChecked(), self.threshold.value())
        self.worker.done.connect(self.analyzed); self.worker.failed.connect(self.failed)
        self.worker.finished.connect(self.finished); self.worker.start()
    def finished(self):
        self.compute.setEnabled(True); self.auto.setEnabled(True)
        self.threshold.setEnabled(not self.auto.isChecked())
        self.apply.setEnabled(self.result is not None)
    def analyzed(self, result):
        self.preview_cache.pop('after', None)
        self.result, count, self.sources, sigma = result
        self.status.setText(f'{count} sampled stars · detection {sigma:g}σ · preview ready')
        self.render()
    def failed(self, message):
        self.result = None
        self.status.setText('Analysis failed; adjust detection threshold and retry.')
        QMessageBox.critical(self, 'White balance', message)
    def render(self, *_):
        key = 'before' if self.before.isChecked() or self.result is None else 'after'
        image = self.image if key == 'before' else self.result
        if key not in self.preview_cache:
            display = np.ascontiguousarray(np.flipud(preview_rgb(image)))
            qim = QImage(display.data, display.shape[1], display.shape[0], display.strides[0], QImage.Format.Format_RGB888)
            qim = qim.scaled(1800, 1800, Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation) if max(image.shape[:2]) > 1800 else qim.copy()
            self.preview_cache[key] = QPixmap.fromImage(qim)
        pixmap = self.preview_cache[key]
        self.item.setPixmap(pixmap)
        for item in list(self.scene.items()):
            if item is not self.item: self.scene.removeItem(item)
        if self.overlay.isChecked() and self.sources is not None:
            pen = QPen(QColor('#00ff80')); pen.setCosmetic(True)
            for star in self.sources:
                x = (float(star['x']) + 0.5) * pixmap.width() / image.shape[1] - 0.5
                y = pixmap.height() - 0.5 - (float(star['y']) + 0.5) * pixmap.height() / image.shape[0]
                self.scene.addEllipse(x-3, y-3, 6, 6, pen)
        self.scene.setSceneRect(self.item.boundingRect())
    def apply_result(self):
        if self.result is None: return
        try:
            with self.siril.image_lock():
                current = self.siril.get_image_pixeldata()
                if current is None or not np.array_equal(current, self.original):
                    raise ValueError('The active image changed. Reopen this script to analyze it again.')
                self.siril.undo_save_state('SAS Star-Based White Balance')
                self.siril.set_image_pixeldata(np.ascontiguousarray(self.result.transpose(2, 0, 1)))
            self.siril.log('SAS star-based white balance applied. Image remains linear; not saved to disk.')
            self.close()
        except Exception as exc: QMessageBox.critical(self, 'Apply failed', str(exc))
    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.status.setText('Please wait for analysis to finish before closing.')
            event.ignore()
        else: event.accept()

def main():
    app = QApplication(sys.argv)
    siril = s.SirilInterface()
    try:
        siril.connect()
        window = WhiteBalanceWindow(siril); window.show(); app.exec()
    except Exception as exc: QMessageBox.critical(None, 'SAS White Balance', str(exc))
if __name__ == '__main__': main()
