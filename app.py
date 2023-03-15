from imageCameraCatch import Camera

INTERVAL = 3
DICTIONARY_PATH = 'dictionary.txt'
MODEL_PATH = 'models/FIRST_MODEL_NEWDATA.h5'


camera = Camera(INTERVAL,MODEL_PATH,DICTIONARY_PATH)
xd = camera.runCameraCapturerApp()


cam = cv2.VideoCapture(0)
cam.release()
