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
        datagen_TRAIN = ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.3,
                        zoom_range=0.3,
                        brightness_range=(0.2, 0.8))
        
        datagen_TEST = ImageDataGenerator(
                        rescale=1./255)
        
        train_generator = datagen_TRAIN.flow_from_directory(
                                            self.train,
                                            target_size=(100,100),
                                            batch_size=self.batch_size,
                                            color_mode = 'grayscale')
        
        valid_generator = datagen_TEST.flow_from_directory(
                                            self.valid,
                                            target_size=(100,100),
                                            batch_size=self.batch_size,
                                            color_mode = 'grayscale')
        
        test_generator = datagen_TEST.flow_from_directory(
                                            self.test,
                                            target_size=(100,100),
                                            batch_size=self.batch_size,
                                            color_mode = 'grayscale')
        
        return train_generator, valid_generator, test_generator



if __name__ == '__main__':
    PATH = 'data'
    TRAIN = 'data/train'
    VALID = 'data/valid'
    TEST = 'data/test'
    data = GeneratorCreator(TRAIN,VALID,TEST,5)
    train_datagen, valid_datagen, test_datagen = data.getGenerators()



