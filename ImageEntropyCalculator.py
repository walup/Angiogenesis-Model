import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


class ImageEntropyCalculator:
    
    #Vamos a suponer que ya viene normalizado el histograma
    def computeEntropy(self, histogram):
        entropy = 0
        for i in range(0,len(histogram)):
            if(histogram[i] != 0):
                entropy = entropy + histogram[i]*np.log(1/histogram[i])
        
        return entropy
    
    
    def computeImageEntropy(self, image):
        values, histogram  = self.extractImageHistogram(image)
        entropy = self.computeEntropy(histogram)
        return entropy
    
    
    def extractImageHistogram(self, image):
        if(len(np.shape(image)) == 3):
            image = np.mean(image, 2)
        
        n = np.size(image, 0)
        m = np.size(image, 1)
        nBins = int(np.floor(np.sqrt(n*m)))
        maxValue = np.max(image)
        minValue = np.min(image)
        binSize = (maxValue - minValue)/nBins
        if(binSize == 0):
            binSize = 1
        
        histogram = np.zeros(nBins)
        values = np.linspace(minValue, maxValue, nBins)
        for i in range(0,n):
            for j in range(0,m):
                value = image[i,j]
                index = int((value - minValue)/binSize)
                if(index == nBins):
                    index = index - 1
                histogram[index] += 1
        histogram = histogram/sum(histogram)
        return values, histogram