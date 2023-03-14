import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import os
import random

"Basic class for opening random image files from dataset folder"

class DataReader:
    def __init__(self, path:str):
        self.path = path
 
    def getRandomItem(self):
        """Returns path to random image from random dir(sign)"""
        dirs = os.listdir(self.path)
        rng_dir = random.randint(0, len(dirs))

        sign = dirs[rng_dir]
        sign_path = self.path + '/' + sign
        sign_photos = os.listdir(sign_path)
        rng_sign = random.randint(0, len(sign_photos))
        photo = sign_photos[rng_sign]
        return sign_path + '/' + photo
    
    def openImageFile(self):
        return plt.imread(self.getRandomItem())
    
    
        
reader = DataReader('main_data')

image = reader.openImageFile()        

plt.imshow(image)        
        
        
        
        
        
        