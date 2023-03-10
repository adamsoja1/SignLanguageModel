import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator


class GeneratorCreator:
    def __init__(self,train,valid,test,batch_size):
        self.train = train
        self.valid = valid
        self.test = test
        self.batch_size = batch_size
        
    def getGenerators(self):
        datagen = ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True)
        
        train_generator = datagen.flow_from_directory(
                                            self.train,
                                            target_size=(200,200),
                                            batch_size=self.batch_size)
        
        valid_generator = datagen.flow_from_directory(
                                            self.valid,
                                            target_size=(200,200),
                                            batch_size=self.batch_size)
        
        test_generator = datagen.flow_from_directory(
                                            self.test,
                                            target_size=(200,200),
                                            batch_size=self.batch_size)
        
        return train_generator, valid_generator, test_generator




PATH = 'data'
TRAIN = 'data/train'
VALID = 'data/valid'
TEST = 'data/test'
data = GeneratorCreator(TRAIN,VALID,TEST,5)
train_datagen, valid_datagen, test_datagen = data.getGenerators()

