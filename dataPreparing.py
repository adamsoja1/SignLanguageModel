import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import random


class Preparation:
    def __init__(self, path:str):
        self.path = path
        
        
        
    def splitImages(self,
                    train_size:float,
                    test_size:float,
                    valid_size:float):
        
        value = train_size + test_size + valid_size
        if value!=1:
            raise ValueError('The sum of train size, test size and valid size must be equal to 1!')
        
        train_dir_path, test_dir_path, valid_dir_path = self.prepareDirs()
        
        signs = os.listdir(self.path)
        
        
        "Sign is a folder"
        "photos are the photos in each sign folders"
        for sign in signs:
            photos = os.listdir(f'{self.path}/{sign}')
            random.shuffle(photos)
            
            train = train_size * len(photos)
            train_photos = photos[0:int(train)]
            
            test = test_size * len(photos)
            test_photos = photos[int(train):int(train)+int(test)]
            
            valid = valid_size * len(photos)
            valid_photos = photos[int(test)+int(train):len(photos)]
            
            "iterate each  photo from train,test,valid list elements"
            
            for photo in train_photos:
                source =  f'{self.path}/{sign}/{photo}'
                destination = f'data/train/{sign}'
                shutil.copy2(source,destination)

            for photo in test_photos:
                source = f'{self.path}/{sign}/{photo}'
                destination = f'data/test/{sign}'
                shutil.copy2(source,destination)
                
            for photo in valid_photos:
                source =  f'{self.path}/{sign}/{photo}'
                destination = f'data/valid/{sign}'
                shutil.copy2(source,destination)
                
        return train_photos,test_photos,valid_photos
                
                
        
        
    def prepareDirs(self):
        dirs = self.getDirs()
        base_dir = 'data'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        train_dir  = os.path.join(base_dir,'train')
        test_dir  = os.path.join(base_dir,'test')
        valid_dir  = os.path.join(base_dir,'valid')
        for directory in (train_dir, valid_dir, test_dir):
            if not os.path.exists(directory):
                os.mkdir(directory)
                
        for sign_dir in dirs:
            train_dir_path = os.path.join(train_dir,sign_dir)
            test_dir_path = os.path.join(test_dir,sign_dir)
            valid_dir_path = os.path.join(valid_dir,sign_dir)
            
            for directory in (train_dir_path, test_dir_path, valid_dir_path):
                if not os.path.exists(directory):
                    os.mkdir(directory)
                    
        return train_dir_path, test_dir_path, valid_dir_path
                
            

    
    def getDirs(self):
        return os.listdir(self.path)
        
        
    
    
prep = Preparation('main_data/asl_alphabet_train')    

prep.splitImages(0.6, 0.2, 0.2)

    
    
    
    
    
    
        
"Webcam function to catch video from webcam!"
# import cv2


# webcam=cv2.VideoCapture(0)

# while True:
#     ret,frame=webcam.read()

#     if ret==True:
#         cv2.imshow("Koolac",frame)
#         key=cv2.waitKey(20) & 0xFF
#         if key==ord("q"):
#             break

# webcam.release()
# cv2.destroyAllWindows()
