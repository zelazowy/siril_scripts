"""Align a few mono/RGB FITS stacks; run from Siril's Python menu.

Version 1.0.0. Also runs standalone when dependencies are installed.
Star matching estimates translation, rotation and uniform scale. Pixels are
resampled once with linear interpolation, without intensity normalization.
"""
import sys
from pathlib import Path

import numpy as np

VERSION = "1.0.3"


def read_stack(path):
    from astropy.io import fits
    with fits.open(path, memmap=False) as hdus:
        hdu = next((h for h in hdus if h.data is not None
                    and h.data.ndim in (2, 3)), None)
        if hdu is None:
            raise ValueError(f"No image in {path}")
        data = np.array(hdu.data, dtype=np.float32)
        header = hdu.header.copy()
    if data.ndim == 3 and data.shape[0] not in (1, 3):
        raise ValueError("Expected mono or channel-first RGB FITS data.")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite pixels in {path}; repair them first.")
    return data, header


def mono(data):
    return data.mean(axis=0) if data.ndim == 3 else data


def preview_pixels(data, common, step=1):
    """Stretch for display using shared coverage, excluding padded borders."""
    sample = mono(data)[::step, ::step]
    valid = common[::step, ::step] & np.isfinite(sample)
    values = sample[valid].astype(np.float64)
    if not values.size:
        raise ValueError("No valid pixels for the preview stretch.")
    lo, hi = np.percentile(values, (1, 99.8))
    if hi <= lo:
        return np.zeros(sample.shape, dtype=np.uint8)
    normalized = np.clip((sample.astype(np.float64)-lo)/(hi-lo), 0, 1)
    # Solve the midtones curve so the valid-area median displays at 25%.
    # Unlike a fixed asinh curve, bright stars cannot darken the background.
    median = np.clip((np.median(values)-lo)/(hi-lo), 1e-6, 1-1e-6)
    target = .25
    midtone = median * (target-1) / (2*target*median-target-median)
    stretched = ((midtone-1)*normalized /
                 ((2*midtone-1)*normalized-midtone))
    return np.ascontiguousarray(stretched * 255, dtype=np.uint8)


def warp(data, matrix, shape):
    from scipy.ndimage import affine_transform
    inv = np.linalg.inv(matrix)
    # Astroalign uses (x,y); scipy uses (row,column).
    linear = inv[:2, :2][::-1, ::-1]
    offset = inv[:2, 2][::-1]
    planes = data[None] if data.ndim == 2 else data
    result = np.stack([affine_transform(p, linear, offset, output_shape=shape,
                        order=1, mode="constant", cval=0, prefilter=False)
                       for p in planes])
    # Geometry, rather than pixel brightness, defines valid coverage.
    coverage = affine_transform(np.ones(planes.shape[1:], np.uint8), linear,
                                offset, output_shape=shape, order=1,
                                mode="constant", cval=0, prefilter=False) == 1
    return (result[0] if data.ndim == 2 else result), coverage


def largest_rectangle(mask):
    """Largest axis-aligned all-valid rectangle, as x,y,width,height."""
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, 0, 0, 0)
    area = 0
    for y, row in enumerate(mask):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for x in range(len(heights) + 1):
            height = int(heights[x]) if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > height:
                left, h = stack.pop()
                if h * (x - left) > area:
                    area = h * (x - left)
                    best = (left, y - h + 1, x - left, h)
                start = left
            if not stack or stack[-1][1] < height:
                stack.append((start, height))
    if not area:
        raise ValueError("Images have no common rectangular coverage.")
    return best


def align(paths, report=lambda message: None):
    import astroalign
    reference, ref_header = read_stack(paths[0])
    target = mono(reference)
    images = [(reference, ref_header)]
    common = np.ones(target.shape, bool)
    for path in paths[1:]:
        report(f"Matching stars: {Path(path).name}")
        data, header = read_stack(path)
        transform, (source_stars, target_stars) = astroalign.find_transform(
            mono(data), target, max_control_points=100)
        residual = np.linalg.norm(transform(source_stars) - target_stars, axis=1)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if len(residual) < 3 or rms > 2:
            raise ValueError(f"Unreliable match for {path}: RMS {rms:.2f} px.")
        report(f"{len(residual)} matched stars; RMS {rms:.3f} px; "
               f"rotation {np.degrees(transform.rotation):.3f}°; "
               f"scale {transform.scale:.6f}")
        output, valid = warp(data, transform.params, target.shape)
        common &= valid
        images.append((output, header))
    return images, common


def output_header(original, reference, crop):
    import re
    header = original.copy()
    # Drop stale astrometry from the moving frame; copy the reference grid.
    pattern = re.compile(r"^(WCSAXES|CTYPE\d+[A-Z]?|CUNIT\d+[A-Z]?|CRPIX\d+[A-Z]?|"
                         r"CRVAL\d+[A-Z]?|CDELT\d+[A-Z]?|CROTA\d+[A-Z]?|"
                         r"(?:CD|PC|PV|PS)\d+_\d+[A-Z]?|LONPOLE[A-Z]?|LATPOLE[A-Z]?|"
                         r"RADESYS[A-Z]?|EQUINOX[A-Z]?|[AB]P?_ORDER|[AB]P?_\d+_\d+)$")
    for key in list(header):
        if pattern.match(key) or key in ("BSCALE", "BZERO", "BLANK", "CHECKSUM", "DATASUM"):
            del header[key]
    for card in reference.cards:
        if pattern.match(card.keyword):
            header[card.keyword] = (card.value, card.comment)
    x, y, _, _ = crop
    for key in list(header):
        if re.fullmatch(r"CRPIX1[A-Z]?", key):
            header[key] -= x
        elif re.fullmatch(r"CRPIX2[A-Z]?", key):
            header[key] -= y
    header.add_history(f"Align Stacks {VERSION}: reference grid; linear interpolation.")
    header.add_history(f"Common crop (zero-based FITS array): {crop}")
    return header


def save_outputs(paths, images, crop, suffix):
    from astropy.io import fits
    if not suffix or any(c in suffix for c in '/\\'):
        raise ValueError("Enter a non-empty suffix without path separators.")
    outputs = [Path(p).with_name(Path(p).stem + suffix + ".fit") for p in paths]
    if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
        raise ValueError("An output already exists or output names collide. Change the suffix.")
    x, y, w, h = crop
    if min(x, y) < 0 or min(w, h) <= 0:
        raise ValueError("Invalid crop.")
    for data, _ in images:
        if x + w > data.shape[-1] or y + h > data.shape[-2]:
            raise ValueError("Crop exceeds image dimensions.")
    for path, (data, header) in zip(outputs, images):
        fits.PrimaryHDU(data=data[..., y:y+h, x:x+w],
                        header=output_header(header, images[0][1], crop)).writeto(path)
    return outputs


def main():
    try:
        import sirilpy as s
    except ImportError:
        s = None
    if s is not None:
        for package in ("PyQt6", "astropy", "scipy", "astroalign"):
            s.ensure_installed(package)
    from PyQt6 import QtCore, QtGui, QtWidgets as Q
    app = Q.QApplication.instance() or Q.QApplication(sys.argv)

    class Preview(Q.QGraphicsView):
        def __init__(self):
            super().__init__()
            self.scene_ = Q.QGraphicsScene(self)
            self.setScene(self.scene_)
            self.setMinimumSize(650, 400)
            self.origin = None
            self.box = None
            self.crop = None

        def show_image(self, data, crop, common):
            image = mono(data)
            step = max(1, int(np.ceil(max(image.shape) / 1800)))
            pixels = preview_pixels(data, common, step)
            qimage = QtGui.QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                                 pixels.strides[0], QtGui.QImage.Format.Format_Grayscale8).copy()
            self.scene_.clear()
            item = self.scene_.addPixmap(QtGui.QPixmap.fromImage(qimage))
            item.setScale(step)
            self.scene_.setSceneRect(0, 0, image.shape[1], image.shape[0])
            self.box = self.scene_.addRect(QtCore.QRectF(), QtGui.QPen(QtCore.Qt.GlobalColor.green, 0))
            self.set_crop(crop)
            self.fitInView(self.scene_.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        def set_crop(self, crop):
            self.crop = crop
            if self.box:
                self.box.setRect(QtCore.QRectF(*crop))

        def mousePressEvent(self, event):
            if self.box and event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.origin = self.mapToScene(event.position().toPoint())

        def mouseMoveEvent(self, event):
            if self.origin is not None:
                rect = QtCore.QRectF(self.origin, self.mapToScene(event.position().toPoint())).normalized()
                rect = rect.intersected(self.scene_.sceneRect())
                x, y = int(np.ceil(rect.x())), int(np.ceil(rect.y()))
                self.set_crop((x, y, max(0, int(rect.right())-x), max(0, int(rect.bottom())-y)))

        def mouseReleaseEvent(self, event):
            self.origin = None

    class Worker(QtCore.QThread):
        message = QtCore.pyqtSignal(str)
        result = QtCore.pyqtSignal(object)
        error = QtCore.pyqtSignal(str)

        def __init__(self, paths):
            super().__init__()
            self.paths = paths

        def run(self):
            try:
                self.result.emit(align(self.paths, self.message.emit))
            except Exception as exc:
                self.error.emit(str(exc))

    class Dialog(Q.QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"Align Stacks — {VERSION}")
            self.images = None
            self.worker = None
            layout = Q.QVBoxLayout(self)
            layout.addWidget(Q.QLabel("Add FITS stacks. The first image is the reference.\n"
                                     "Optionally use Make reference to change it. Delete/Backspace removes a selected stack."))
            self.files = Q.QListWidget()
            self.files.setMaximumHeight(110)
            self.delete_shortcuts = []
            for key in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Delete):
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.files)
                shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
                shortcut.activated.connect(self.remove_file)
                self.delete_shortcuts.append(shortcut)
            layout.addWidget(self.files)
            row = Q.QHBoxLayout()
            self.add = Q.QPushButton("Add images…")
            self.reference = Q.QPushButton("Make reference")
            self.run = Q.QPushButton("Align / preview")
            for button in (self.add, self.reference, self.run):
                row.addWidget(button)
            layout.addLayout(row)
            self.preview = Preview()
            layout.addWidget(self.preview)
            layout.addWidget(Q.QLabel("Drag a rectangle for a custom shared crop. Preview stretch is display only."))
            row = Q.QHBoxLayout()
            self.auto = Q.QPushButton("Automatic common crop")
            self.choice = Q.QComboBox()
            self.choice.currentIndexChanged.connect(self.display)
            self.suffix = Q.QLineEdit("_aligned")
            self.suffix.setMaximumWidth(140)
            self.save = Q.QPushButton("Save aligned FITS")
            for widget in (self.auto, self.choice, Q.QLabel("Suffix:"), self.suffix, self.save):
                row.addWidget(widget)
            layout.addLayout(row)
            self.log = Q.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(100)
            layout.addWidget(self.log)
            self.add.clicked.connect(self.add_files)
            self.reference.clicked.connect(self.make_reference)
            self.run.clicked.connect(self.start)
            self.auto.clicked.connect(lambda: self.preview.set_crop(self.auto_crop))
            self.save.clicked.connect(self.write)
            self.invalidate()

        def invalidate(self):
            self.images = None
            self.save.setEnabled(False)
            self.auto.setEnabled(False)
            self.choice.clear()
            self.preview.scene_.clear()
            self.preview.box = None

        def add_files(self):
            paths, _ = Q.QFileDialog.getOpenFileNames(self, "Choose stacks", "", "FITS (*.fit *.fits *.fts)")
            existing = {self.files.item(i).text() for i in range(self.files.count())}
            for path in paths:
                if path not in existing:
                    self.files.addItem(path)
                    existing.add(path)
            self.invalidate()

        def remove_file(self):
            index = self.files.currentRow()
            if index >= 0 and self.files.isEnabled():
                self.files.takeItem(index)
                self.invalidate()

        def make_reference(self):
            index = self.files.currentRow()
            if index >= 0:
                self.files.insertItem(0, self.files.takeItem(index))
                self.files.setCurrentRow(0)
                self.invalidate()

        def busy(self, value):
            for widget in (self.files, self.add, self.reference, self.run):
                widget.setEnabled(not value)

        def start(self):
            self.paths = [self.files.item(i).text() for i in range(self.files.count())]
            if len(self.paths) < 2:
                Q.QMessageBox.warning(self, "Align", "Choose at least two stacks.")
                return
            self.invalidate()
            self.busy(True)
            self.log.clear()
            self.worker = Worker(self.paths)
            self.worker.message.connect(self.log.appendPlainText)
            self.worker.result.connect(self.ready)
            self.worker.error.connect(self.failed)
            self.worker.finished.connect(lambda: self.busy(False))
            self.worker.start()

        def ready(self, result):
            try:
                self.images, self.common = result
                self.auto_crop = largest_rectangle(self.common)
                self.preview.crop = self.auto_crop
                self.choice.addItems([Path(p).name for p in self.paths])
                self.save.setEnabled(True)
                self.auto.setEnabled(True)
                self.log.appendPlainText(f"Ready. Automatic crop: {self.auto_crop}")
            except Exception as exc:
                self.failed(str(exc))

        def display(self, index):
            if self.images is not None and index >= 0:
                self.preview.show_image(self.images[index][0], self.preview.crop, self.common)

        def failed(self, message):
            self.invalidate()
            Q.QMessageBox.critical(self, "Alignment failed", message)

        def write(self):
            try:
                x, y, w, h = self.preview.crop
                if w <= 0 or h <= 0 or not self.common[y:y+h, x:x+w].all():
                    raise ValueError("Crop includes uncovered edges. Use Automatic common crop or draw a smaller rectangle.")
                paths = save_outputs(self.paths, self.images, self.preview.crop, self.suffix.text())
                self.log.appendPlainText("Saved:\n" + "\n".join(map(str, paths)))
                Q.QMessageBox.information(self, "Saved", "\n".join(map(str, paths)))
            except Exception as exc:
                Q.QMessageBox.critical(self, "Save failed", str(exc))

        def reject(self):
            if self.worker is None or not self.worker.isRunning():
                super().reject()

        def closeEvent(self, event):
            if self.worker is not None and self.worker.isRunning():
                event.ignore()
            else:
                event.accept()

    dialog = Dialog()
    dialog.exec()


if __name__ == "__main__":
    main()
