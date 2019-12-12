from PyQt5.QtWidgets import QWidget, QTableWidgetItem
from PyQt5 import uic


class WindowSignatures(QWidget):

    def __init__(self, parent=None):
        super(WindowSignatures, self).__init__(parent)
        uic.loadUi('GUI/windowsignatures.ui', self)

        self.table.setColumnCount(4)
        self.table.setRowCount(1)
        self.table.setHorizontalHeaderLabels(
            ["Id", "Pixels number", "Threshold", "Signature"])

    def setSignatures(self, signatures):
        self.signatures = signatures
        self._printSignatures()

    def _printSignatures(self):
        if self.signatures:
            sigNumber = self.signatures[0].shape[0]
            self.table.setRowCount(sigNumber)
            for i in range(sigNumber):
                pixNumber = int(self.signatures[1][i])
                threshold = float(self.signatures[2][i])
                signature = self.signatures[0][i]
                self.table.setItem(i, 0, QTableWidgetItem(str(i)))
                self.table.setItem(i, 1, QTableWidgetItem(str(pixNumber)))
                self.table.setItem(i, 2, QTableWidgetItem(str(threshold)))
                self.table.setItem(i, 3, QTableWidgetItem(str(signature)))
            self.table.resizeColumnsToContents()
