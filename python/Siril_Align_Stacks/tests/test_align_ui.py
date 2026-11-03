"""Offscreen checks for the default workflow and state changes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import unittest
from unittest.mock import patch
from PyQt6 import QtWidgets as Q
import Siril_Align_Stacks as module


class InterfaceTests(unittest.TestCase):
    def test_defaults_modes_and_file_actions(self):
        app = Q.QApplication.instance() or Q.QApplication([])
        def inspect(dialog):
            self.assertFalse(dialog.mode.isChecked())
            self.assertTrue(dialog.settings_button.isHidden())
            self.assertFalse(dialog.run.isEnabled())
            self.assertFalse(dialog.save.isEnabled())
            with patch.object(Q.QFileDialog, 'getOpenFileNames', return_value=(['/tmp/a.fit', '/tmp/b.tif', '/tmp/a.fit'], '')):
                dialog.add_files()
            self.assertEqual(dialog.files.count(), 2)
            self.assertEqual(dialog.files.item(0).text(), 'a.fit')
            self.assertEqual(dialog.files.item(0).toolTip(), '/tmp/a.fit')
            self.assertTrue(dialog.run.isEnabled())
            dialog.files.setCurrentRow(1)
            self.assertTrue(dialog.reference.isEnabled())
            dialog.make_reference()
            self.assertEqual(dialog.files.item(0).text(), 'b.tif')
            self.assertFalse(dialog.reference.isEnabled())
            dialog.mode.setChecked(True)
            self.assertFalse(dialog.settings_button.isHidden())
            self.assertEqual(dialog.suffix.text(), '_aligned_sip')
            dialog.suffix.setText('_custom')
            dialog.mode.setChecked(False)
            self.assertEqual(dialog.suffix.text(), '_custom')
            dialog.busy(True)
            self.assertFalse(dialog.run.isEnabled())
            self.assertFalse(dialog.files.isEnabled())
            dialog.busy(False)
            dialog.remove_file()
            self.assertEqual(dialog.files.count(), 1)
            self.assertFalse(dialog.run.isEnabled())
            with patch.object(Q.QMessageBox, 'critical'):
                dialog.failed('Test failure')
            self.assertFalse(dialog.save.isEnabled())
            self.assertTrue(dialog.details.isChecked())
            self.assertIn('failed', dialog.status.text())
            return 0
        with patch.dict(sys.modules, {'sirilpy': None}), patch.object(Q.QDialog, 'exec', inspect):
            module.main()



    def test_crop_move_resize_redraw_and_window_resize(self):
        import numpy as np
        from PyQt6 import QtCore, QtTest
        app = Q.QApplication.instance() or Q.QApplication([])
        def inspect(dialog):
            dialog.show()
            app.processEvents()
            preview = dialog.preview
            preview.show_image(np.ones((400, 600), np.float32), (100, 100, 200, 150),
                               np.ones((400, 600), bool))
            self.assertTrue(dialog.windowFlags() & QtCore.Qt.WindowType.WindowMaximizeButtonHint)
            def drag(start, end):
                a = preview.mapFromScene(QtCore.QPointF(*start))
                b = preview.mapFromScene(QtCore.QPointF(*end))
                QtTest.QTest.mousePress(preview.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=a)
                QtTest.QTest.mouseMove(preview.viewport(), b)
                QtTest.QTest.mouseRelease(preview.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=b)
            drag((200, 170), (230, 190))
            x,y,w,h = preview.crop
            self.assertAlmostEqual(x, 130, delta=2)
            self.assertAlmostEqual(y, 120, delta=2)
            self.assertEqual((w,h), (200,150))
            drag((x+w,y+h), (x+w+40,y+h+30))
            self.assertAlmostEqual(preview.crop[2], 240, delta=2)
            self.assertAlmostEqual(preview.crop[3], 180, delta=2)
            drag((450,300), (550,380))
            self.assertAlmostEqual(preview.crop[0], 450, delta=2)
            self.assertAlmostEqual(preview.crop[2], 100, delta=2)
            before = preview.crop
            scale = preview.transform().m11()
            dialog.resize(dialog.width()+300, dialog.height()+250)
            app.processEvents()
            self.assertGreater(preview.transform().m11(), scale)
            self.assertEqual(preview.crop, before)
            # Move to the image boundary while preserving crop dimensions.
            drag((500,340), (599,399))
            x,y,w,h = preview.crop
            self.assertLessEqual(x+w,600)
            self.assertLessEqual(y+h,400)
            self.assertEqual(len(preview.handles),8)
            dialog.close()
            return 0
        with patch.dict(sys.modules, {'sirilpy': None}), patch.object(Q.QDialog, 'exec', inspect):
            module.main()

if __name__ == '__main__':
    unittest.main()
