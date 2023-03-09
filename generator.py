import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator


DATA = 'asl_dataset' 


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        DATA,
        target_size=(150, 150),
        batch_size=5)


x,y = next(train_generator)

y = np.argmax(y,axis=-1)
x = np.array(x)

for i in range(5):
    plt.imshow(x[i])
    plt.show()
    


