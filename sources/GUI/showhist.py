import numpy as np
import pyqtgraph as pg
import sys

from PyQt5.QtWidgets import QWidget, QTableWidgetItem
from PyQt5 import uic

class Show_hist(QWidget):

    def __init__(self):
        super(Show_hist, self).__init__()
        uic.loadUi('GUI/showhist.ui', self)
        self.graphWidget.setBackground('w')

    def plot(self, mask_values, mask_hsi, num_cust):
        vals = mask_values[mask_hsi == num_cust]
        y, x = np.histogram(vals, bins=np.linspace(np.min(vals), np.max(vals), 100))
        self.graphWidget.plot( x, y, stepMode = True, fillLevel = 0, brush = (0,0,255,150) )
        
        #y = pg.pseudoScatter(vals, spacing=0.15)
        #self.graphWidget.plot(vals, y, pen=None, symbol='o', symbolSize=5, symbolPen=(255,255,255,200), symbolBrush=(0,0,255,150))