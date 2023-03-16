import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import ast
import cv2

MODEL_PATH = 'models/FIRST_MODEL_NEWDATA.h5'
model = tf.keras.models.load_model(MODEL_PATH)

with open('dictionary.txt') as f:
    data = f.read()
dictionary = ast.literal_eval(data)


test_path = 'data/research'
test_photos = os.listdir(test_path)

def preprocess(image):
    image = cv2.resize(image,(100,100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1,100,100,1)
    image = image/255
    return image

for photo in range(len(test_photos)):
    image = plt.imread(f'{test_path}/{test_photos[photo]}')
    image = preprocess(image)
    y_prop = model.predict(image)
    y_predict = np.argmax(y_prop,axis=-1)
    y_predict = y_predict[0]
  
 
    print(f'Photo {test_photos[photo]} predicted as: {dictionary[y_predict]}')
    


