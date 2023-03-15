import cv2 
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Camera:

    
    def __init__(self,interval):
        self.interval = interval
    

    def cameraReader(self):
        return cv2.VideoCapture(0)

    def runCameraCapturer(self):
        camera = self.cameraReader()
        global image
        while True:
            capture, frame = camera.read()
            if capture:
                img = np.asarray(frame)
                image = self.imagePreprocess(img)
                
                
                

            plt.imshow(image,cmap='gray')
            plt.show()
            time.sleep(self.interval)
        

    def imagePreprocess(self,image):
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image,(100,100))
        return image

    def getModel(self,path):
        model = tf.load_model(path)
        return model

    def predictSign(self):
        pass
            



