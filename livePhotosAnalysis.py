import cv2 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

image  = plt.imread('data/research/4.jpg')
image = cv2.resize(image,(200,200))

plt.imshow(image)


# def increase_brightness(img, value=30):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     lim = 255 - value
#     v[v > lim] = 255
#     v[v <= lim] += value

#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img


# image = increase_brightness(image, value=50)


sample_image = cv2.imread('data/research/4.jpg')
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(200,200))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# _,threshInv = cv2.threshold(gray, 130, 255,
# 	cv2.THRESH_BINARY_INV)

otsu_threshold, image_result = cv2.threshold(
    gray, 0, 1, cv2.THRESH_OTSU,
)

new = gray * image_result

plt.imshow(np.array(new),cmap='gray')




def handOutput(image,mask):
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    image[:,:,0] = R*mask
    image[:,:,1] = B*mask
    image[:,:,2] = G*mask
    return image

out = handOutput(image,image_result)

out = np.array(out)
out = out/255

plt.imshow(out)
out = out.reshape(1,200,200,3)
model = tf.keras.models.load_model('Third_model_100epochs.h5',compile=False)
image = image.reshape(1,200,200,3)
image = image/255
prediction = model.predict(out)

pr = np.argmax(prediction)

from dictionary import Dictionary
dct = Dictionary('notsplitted')
translator = dct.createDictionary()
translator[16]









