import cv2 
import time
import numpy as np
import tensorflow as tf
import ast
import matplotlib.pyplot as plt
class Camera:

    def __init__(self,interval:int,model_path:str,dictionary_path:str):
        self.model_path = model_path
        self.interval = interval
        self.dictionary_path = dictionary_path
        
        "initiate methods"
        self.getModel(self.model_path)
        self.translator()
        
        
    def cameraReader(self):
        return cv2.VideoCapture(0)

    def runCameraCapturerApp(self):
        camera = self.cameraReader()
        while True:
            capture, frame = camera.read()
            if capture:
                img = np.asarray(frame)
                image = self.imagePreprocess(img)
                result = self.predictSign(image)   
            
          
            
            print(self.translator[result])                           
            time.sleep(self.interval)
        

    def imagePreprocess(self,image:int)->float:
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image,(100,100))
        image = image/255
        image = image.reshape(100,100,1)
        return image

    def getModel(self,path):
        self.model = tf.keras.models.load_model(path)
        return self.model

    def predictSign(self,image):
        predict = self.model.predict(image)
        predict = np.argmax(predict,axis=-1)
        return predict
            
    def translator(self):
        with open(self.dictionary_path) as f:
            data = f.read()
        dictionary = ast.literal_eval(data)
        self.translator= dictionary
        

        
        
            
            


