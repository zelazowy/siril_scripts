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


if __name__ == '__main__':
    unittest.main()
