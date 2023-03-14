import cv2 
import time
import numpy as np
import tensorflow as tf

class Camera:
    def __init__(self,interval):
        self.interval = interval
    

    def cameraReader(self):
        return cv2.VideoCapture(0)

    def runCameraCapturer(self):
        camera = self.cameraReader()
        while True:
            capture, frame = camera.read()
            if capture:
                image = np.asarray(frame)
                
            
        time.sleep(self.interval)

    def imagePreprocess(self,image):
        if image.shape[-1] == 3:
            image = cv2.COLOR_RGB2GRAY(image)
        image = cv2.resize(image,(100,100))
        return image

    def getModel(self,path):
        model = tf.load_model(path)
        return model

    def predictSign(self):
        pass
            


