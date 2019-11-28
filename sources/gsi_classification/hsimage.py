import numpy as np
import qimage2ndarray

class HSImage():

    def __init__(self, data):
        self.dataArray = data

    def getLayerAsQImage(self, layerID = 0):
        img = self.dataArray[:, :, layerID]
        img = img - np.min(img)
        if np.max(img) != 0.0:
            img = img // (np.max(img) / 255.0)
        return qimage2ndarray.array2qimage(img)

    def getRgbImage(self, channels = (35, 20, 7)):
        img = self.dataArray[:, :, channels]
        img = img - np.min(img)
        if np.max(img) != 0.0:
            img = img // (np.max(img) / 255.0)
        return qimage2ndarray.array2qimage(img)

    def getNumpyRgbImage(self, channels = (35, 20, 7)):
        img = self.dataArray[:, :, channels]
        return img