from model import resnet50
import tensorflow as tf
from generator import GeneratorCreator
from dictionary import Dictionary
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np 
from plots import make_plot
import seaborn as sns
import matplotlib.pyplot as plt
import os


NAME_EXPERIMENT = 'FIRST_MODEL_NEWDATA'
PATH = 'data'
TRAIN = 'data/train'
VALID = 'data/valid'
TEST = 'data/test'
BATCH_SIZE = 30
data = GeneratorCreator(TRAIN,VALID,TEST,BATCH_SIZE)
train_datagen, valid_datagen, test_datagen = data.getGenerators()


WIDTH = 100
HEIGHT = 100
CHANNELS = 1
NUM_CLASSES = len(os.listdir(TRAIN))
OPTIMIZER  = tf.keras.optimizers.Adam(lr=0.00007)
METRICS = ['accuracy',]
LOSS = 'categorical_crossentropy'

model = resnet50(NUM_CLASSES, WIDTH, HEIGHT, CHANNELS)
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS])

history = model.fit(
            train_datagen,
            epochs = 100,
            validation_data=valid_datagen,
            )


y_test = []
y_pred = []

rang = int(500/25)

for i in range(rang):
    x_test,y_t = next(test_datagen)
    batch_ytest = np.argmax(y_t,axis=-1)
    
    for i in batch_ytest:
        y_test.append(i)
    
    y_pr = model.predict(x_test)
    batch_ypred = np.argmax(y_pr,axis=-1)
    for i in batch_ypred:
        y_pred.append(i)
   





conf = confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
np.set_printoptions(threshold=np.inf)

file = open(f"results/{NAME_EXPERIMENT}_report.txt","w")     
file.write(classification_report(y_test, y_pred))
file.close()

file = open(f"results/{NAME_EXPERIMENT}_matrix.txt","w")     
file.write(str(conf))
file.close()



img = sns.heatmap(conf)
fig = img.get_figure()
fig.savefig("results/{NAME_EXPERIMENT}_mat.png")

model.save(f'{NAME_EXPERIMENT}.h5')



make_plot(history,'accuracy','first_model')
make_plot(history,'loss','first_model_loss')