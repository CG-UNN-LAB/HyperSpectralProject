import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt
from PyQt5 import uic
from GUI.imageviewer import ImageViewer
from gsi_classification.hsimage import HSImage
import os
import spectral.io.envi as envi
import gsi_classification.clustering
from GUI.windowsignatures import WindowSignatures
import qimage2ndarray

from gsi_classification.clustering import reference_clustering


class MainWindow(QMainWindow):

    def __init__(self, *args, inputFile=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('GUI/mainwindow.ui', self)
        self.imageviewer = ImageViewer()
        self.viewerLayout.addWidget(self.imageviewer)
        self.signatures = []

        self.imageviewer.midMouseButtonPressed.connect(self._addCluster)
        self.actionOpen_file.triggered.connect(self._openImage)
        self.actionSignatures_window.triggered.connect(
            self._openSignaturesWindow)
        self.sliderLayers.valueChanged.connect(self._changeLayer)
        self.buttonRgbImage.clicked.connect(self._showRgbImage)
        self.buttonShowClusters.clicked.connect(self._showClustersImage)

        if inputFile:
            self._loadFile(inputFile)

    def _loadFile(self, fileName):
        if fileName.endswith(('jpg', 'bmp', 'png')):
            self.imageviewer.loadImageFromFile(fileName)
            self.sliderLayers.setMaximum(1)
        elif fileName.endswith(('npy')):
            self.hsimage = HSImage(np.load(fileName))
            self.sliderLayers.setMaximum(self.hsimage.dataArray.shape[2]-1)
            self.imageviewer.setImage(self.hsimage.getLayerAsQImage(0))
        elif fileName.endswith(('hdr')):
            fileName2 = os.path.splitext(fileName)[0] + '.img'
            self.EnviImage = envi.open(fileName, fileName2)
            self.hsimage = HSImage(np.array(self.EnviImage.open_memmap()))
            self.sliderLayers.setMaximum(self.hsimage.dataArray.shape[2] - 1)
            self.imageviewer.setImage(self.hsimage.getLayerAsQImage(0))

    def _openImage(self):
        fileName, dummy = QFileDialog.getOpenFileName(self, "Open image file.")
        self._loadFile(fileName)

    def _changeLayer(self):
        newLayer = self.sliderLayers.value()
        try:
            if self.hsimage:
                self.imageviewer.setImage(
                    self.hsimage.getLayerAsQImage(newLayer))
        except AttributeError:
            print('Not numpy image loaded')

    def _parseRgbValuesFromForm(self):
        r = int(self.lineR.text())
        g = int(self.lineG.text())
        b = int(self.lineB.text())
        return (r, g, b)

    def _showRgbImage(self):
        try:
            channels = self._parseRgbValuesFromForm()
            if self.hsimage:
                self.imageviewer.setImage(
                    self.hsimage.getRgbImage(channels=channels))
        except AttributeError:
            print("Not numpy image loaded or wrong values in textboxes")

    def _addCluster(self, x, y):
        text = self.textBoxClusters.toPlainText()
        newLine = 'Add cluster pos {} {}'.format(int(x), int(y))
        self.textBoxClusters.setText(text + newLine + '\n')

        if self.signatures == []:
            spectr = np.array(self.hsimage.dataArray[int(x), int(y), 0:])
            spectr = np.expand_dims(spectr, axis=0)
            cluster = [spectr, [1], [1.0]]
            self.signatures += cluster
        else:
            spectr = np.array(self.hsimage.dataArray[int(x), int(y), 0:])
            spectr = np.expand_dims(spectr, axis=0)
            spectr = np.concatenate((self.signatures[0], spectr), axis=0)

            pixcounts = self.signatures[1] + [1]
            probs = self.signatures[2] + [1.0]

            self.signatures = [spectr, pixcounts, probs]

    def _showClustersImage(self):

        #spectr = np.array(array[50, 50, 0:224])
        #spectr = np.expand_dims(spectr, axis = 0)
        #cluster = [spectr, [1], [0.95]]

        try:
            threshold = float(self.lineThreshold.text())
        except ValueError:
            print('Wrong value in textbox Threshold')
            return

        rgb_image = self.hsimage.getNumpyRgbImage(channels=(35, 20, 7))
        line_RGB_hsi = np.float64(
            rgb_image.reshape((-1, rgb_image.shape[2]))) * 255

        line_hsi = self.hsimage.dataArray.reshape(
            (-1, self.hsimage.dataArray.shape[2]))
        mask_hsi, color_mask_hsi, cluster_hsi = reference_clustering(
            line_hsi, threshold, [], False, line_RGB_hsi)

        #print("MASK HSI SHAPE = ", mask_hsi.shape)
        #print("COLOR MASK HSI SHAPE = ", color_mask_hsi.shape)
        #print("CLUSTER HSI SHAPE = ", len(cluster_hsi))
        # print(cluster_hsi[0].shape)

        img = color_mask_hsi.reshape(rgb_image.shape)
        img = img - np.min(img)
        if np.max(img) != 0.0:
            img = img // (np.max(img) / 255.0)
        self.imageviewer.setImage(qimage2ndarray.array2qimage(img))
        self.signatures += cluster_hsi

    def _openSignaturesWindow(self):
        self.w = WindowSignatures()
        self.w.setSignatures(self.signatures)
        self.w.show()
