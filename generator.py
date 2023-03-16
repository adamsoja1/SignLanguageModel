import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ast
from keras.preprocessing.image import ImageDataGenerator
import time

class GeneratorCreator:
    def __init__(self,train,valid,test,batch_size):
        self.train = train
        self.valid = valid
        self.test = test
        self.batch_size = batch_size


            
    def getGenerators(self):
        datagen_TRAIN = ImageDataGenerator(
                        rescale=1./255,
                       
                        zoom_range=0.1,
                        brightness_range=(0.2, 0.8))
        
        datagen_TEST = ImageDataGenerator(
                        rescale=1./255)
        
        train_generator = datagen_TRAIN.flow_from_directory(
                                            self.train,
                                            target_size=(300,300),
                                            batch_size=self.batch_size,
                                            color_mode = 'rgb',
                                            class_mode = 'categorical')
        
        valid_generator = datagen_TEST.flow_from_directory(
                                            self.valid,
                                            target_size=(300,300),
                                            batch_size=self.batch_size,
                                            color_mode = 'rgb')
        
        test_generator = datagen_TEST.flow_from_directory(
                                            self.test,
                                            target_size=(100,100),
                                            batch_size=self.batch_size,
                                            color_mode = 'rgb')
        
        return train_generator, valid_generator, test_generator



if __name__ == '__main__':
    PATH = 'data'
    TRAIN = 'data/train'
    VALID = 'data/valid'
    TEST = 'data/test'
    data = GeneratorCreator(TRAIN,VALID,TEST,5)
    train_datagen, valid_datagen, test_datagen = data.getGenerators()

with open('dictionary.txt') as f:
    data = f.read()
dictionary = ast.literal_eval(data)

while True:
    X,Y = next(train_datagen)
    for i in range(X.shape[0]):
        plt.imshow(X[i])
        plt.show()
        xd = np.argmax(Y[i],axis=-1)
      
        print(dictionary[xd])
        time.sleep(1)
        
        
labels = (train_datagen.class_indices)        
labels = dict((v,k) for k,v in labels.items())        
f = open("dictionary.txt","w")
# write file
f.write(str(labels))
# close file
f.close()