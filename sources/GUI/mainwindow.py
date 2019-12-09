import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QMainWindow, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt
from PyQt5 import uic
from GUI.imageviewer import ImageViewer
from gsi_classification.hsimage import HSImage
import os
import spectral.io.envi as envi
import cv2
import gsi_classification.clustering
from GUI.windowsignatures import WindowSignatures
from GUI.showhist import Show_hist
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
        self.imageviewer.leftMouseButtonDoubleClicked.connect(self._separation_cluster)
        self.actionOpen_file.triggered.connect(self._openImage)
        self.actionSignatures_window.triggered.connect(
            self._openSignaturesWindow)
        self.sliderLayers.valueChanged.connect(self._changeLayer)
        self.buttonRgbImage.clicked.connect(self._showRgbImage)
        self.buttonShowClusters.clicked.connect(self._showClustersImage)
        self.checkBox_canny.toggled.connect(self._onClicked_canny)
        self.checkBox_canny.setEnabled(False)
        self.pushButton_bac_to_img.clicked.connect(self._bac_to_img)
        self.class_num = 0
        self.pushButton_show_hist.clicked.connect(self._show_hist)
        self.pushButton_show_hist.setEnabled(False)

        if inputFile:
            self._loadFile(inputFile)

    def _show_hist(self):
        self.h = Show_hist()
        self.h.plot(self.mask_values, self.mask_hsi, self.class_num)
        self.h.show()

    def _bac_to_img(self):
    	self.imageviewer.setImage(qimage2ndarray.array2qimage(self.img))
    	self.buttonShowClusters.setText("Show Clusters image")
    	self.class_num = 0
    	self.label_num_class.setText("Сlass number: ")
    	self.label_num_of_pix.setText("Number of pixels in the class: ")
    	self.pushButton_bac_to_img.setEnabled(False)
    	self.pushButton_show_hist.setEnabled(False)

    def _separation_cluster(self, x, y):
    	x = int(x)
    	y = int(y)
    	self.class_num = self.mask_hsi[ x + y * self.img.shape[1] ]
    	self.label_num_class.setText("Сlass number: " + str( self.class_num ))
    	self.label_num_of_pix.setText("Number of pixels in the class: " + str(int(self.cluster_hsi[1][self.class_num - 1])))
    	color_sep_img = self.color_mask_hsi.copy()
    	#color_sep_img[self.mask_hsi != self.class_num] = [16, 16, 16]
    	color_sep_img[self.mask_hsi != self.class_num] = np.uint8( color_sep_img[self.mask_hsi != self.class_num] * 0.2 )
    	color_sep_img = color_sep_img.reshape(self.img.shape)
    	self.imageviewer.setImage(qimage2ndarray.array2qimage(color_sep_img))
    	self.pushButton_show_hist.setEnabled(True)
    	self.pushButton_bac_to_img.setEnabled(True)
    	self.buttonShowClusters.setText("Separation cluster")

    def _onClicked_canny(self):
        checkBox_canny = self.sender()
        if checkBox_canny.isChecked():
            self.img = np.uint8(self.img)
            canny_pic = cv2.Canny(self.img, 1, 1)
            self.imageviewer.setImage(qimage2ndarray.array2qimage(canny_pic))
        else:
            self.imageviewer.setImage(qimage2ndarray.array2qimage(self.img))

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

                self.label_show_layer.setText("layer: " + str(newLayer))
        
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
        newLine = '{} {}'.format(int(x), int(y))
        self.textBoxClusters.setText(text + newLine + '\n')

    def _showClustersImage(self):
        self.checkBox_canny.setEnabled(True)

        #spectr = np.array(array[50, 50, 0:224])
        #spectr = np.expand_dims(spectr, axis = 0)
        #cluster = [spectr, [1], [0.95]]

        try:
            threshold = float(self.lineThreshold.text())
        except ValueError:
            print('Wrong value in textbox Threshold')
            return

        try:
            rgb_image = self.hsimage.getNumpyRgbImage(channels=(35, 20, 7))
            line_RGB_hsi = np.float64(
                rgb_image.reshape((-1, rgb_image.shape[2]))) * 255
        except AttributeError:
        	print("Not numpy image loaded")
        	return

        line_hsi = self.hsimage.dataArray.reshape(
            (-1, self.hsimage.dataArray.shape[2]))

        metric = self.comboBox_Metric_selection.currentText()

        if self.class_num == 0:
        	self.mask_hsi, self.color_mask_hsi, self.cluster_hsi, self.mask_values = reference_clustering(
            	line_hsi, threshold, [], True, metric)#, line_RGB_hsi)
        else:

        	sep_line_hsi = line_hsi.copy()
        	sep_line_hsi[self.mask_hsi != self.class_num] = 0

        	self.mask_hsi, self.color_mask_hsi, self.cluster_hsi, self.mask_values = reference_clustering(
            	sep_line_hsi, threshold, [], True, metric)#, line_RGB_hsi)

        #print("MASK HSI SHAPE = ", mask_hsi.shape)
        #print("COLOR MASK HSI SHAPE = ", color_mask_hsi.shape)
        #print("CLUSTER HSI SHAPE = ", len(cluster_hsi))
        #print(cluster_hsi[0].shape)
        #print('kek',  ImageViewer.)
        
        self.label_quan_of_class.setText("Quantity of classes: " + str(self.cluster_hsi[0].shape[0])) 

        self.img = self.color_mask_hsi.reshape(rgb_image.shape)
        self.img = self.img - np.min(self.img)
        if np.max(self.img) != 0.0:
            self.img = self.img // (np.max(self.img) / 255.0)
        self.imageviewer.setImage(qimage2ndarray.array2qimage(self.img))
        self.signatures += self.cluster_hsi

    def _openSignaturesWindow(self):
        self.w = WindowSignatures()
        self.w.setSignatures(self.signatures)
        self.w.show()
