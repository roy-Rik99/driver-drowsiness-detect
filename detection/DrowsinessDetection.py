import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random   ## shuffling data to avoid overfitting
import pickle
from tensorflow import keras
from tensorflow.keras import layers
import winsound
import numpy as np

img_array = cv2.imread("Test_Dataset\Closed_Eyes\s0001_00001_0_0_0_0_0_01.png",cv2.IMREAD_GRAYSCALE)

plt.imshow(img_array,cmap="gray")

img_array.shape

Datadirectory = "Test_Dataset/" ## training dataset
Classes = ["Closed_Eyes","Open_Eyes"] ## list of classes
for category in Classes:
    path = os.path.join(Datadirectory,category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break

img_size = 224 ## resizing the image from 86 x 86 to 224 X 224
new_array = cv2.resize(backtorgb,(img_size,img_size))
plt.imshow(new_array,cmap = "gray")
plt.show()


# ## Reading all the images and converting them into an array for data and labels

training_Data = []
def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory,category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
                new_array = cv2.resize(backtorgb,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass


create_training_Data()

print(len(training_Data)) ## displaying length of training_Data

random.shuffle(training_Data)

X = []
y = []
for features,label in training_Data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)

X.shape

X = X/255.0; # we are normalizing it

y = np.array(y)

pickle_out = open("X_pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y_pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

pickle_in = open("X_pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_pickle","rb")
y = pickle.load(pickle_in)

model = tf.keras.applications.mobilenet.MobileNet()

model.summary()

base_input = model.layers[0].input ## input

base_output = model.layers[-4].output ## output

Flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer) ## one node (1/0)
final_output = layers.Activation('sigmoid')(final_output)
new_model = keras.Model(inputs = base_input,outputs = final_output)

new_model.summary()

new_model.compile(loss = "binary_crossentropy",optimizer = "adam", metrics = ["accuracy"])

new_model.fit(X,y,epochs = 1,validation_split = 0.1) ## training

new_model.save('my_model.h5')

new_model = tf.keras.models.load_model('my_model.h5')

img_array = cv2.imread("Test_Dataset\Open_Eyes\s0001_01871_0_0_1_0_0_01.png",cv2.IMREAD_GRAYSCALE)
backtorgb = cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
new_array = cv2.resize(backtorgb,(img_size,img_size))

X_input = np.array(new_array).reshape(1,img_size,img_size,3)

X_input.shape

plt.imshow(new_array)

X_input = X_input/255.0

prediction = new_model.predict(X_input)

prediction

img_array2 = cv2.imread("Test_Dataset\Closed_Eyes\s0001_00114_0_0_0_0_0_01.png",cv2.IMREAD_GRAYSCALE)
backtorgb2 = cv2.cvtColor(img_array2,cv2.COLOR_GRAY2RGB)
new_array2 = cv2.resize(backtorgb2,(img_size,img_size))

X_input2 = np.array(new_array2).reshape(1,img_size,img_size,3)

X_input2.shape

plt.imshow(new_array2)

X_input2 = X_input2/255.0
prediction = new_model.predict(X_input2)
prediction

img = cv2.imread("testimg.jpg")

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(gray,1.1,4)

faces = faceCascade.detectMultiScale(img,1.1,5)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

for(x,y,w,h) in eyes:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
eyes = eye_cascade.detectMultiScale(gray,1.1,4)
for x,y,w,h in eyes:
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyess = eye_cascade.detectMultiScale(roi_gray)
    if len(eyess) == 0:
        print("eyes are not detected")
    else:
        for (ex,ey,ew,eh) in eyess:
            eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]

plt.imshow(cv2.cvtColor(eyes_roi,cv2.COLOR_BGR2RGB))

eyes_roi.shape

final_image = cv2.resize(eyes_roi,(224,224))
final_image = np.expand_dims(final_image,axis=0) ##need fourth dimension
final_image = final_image/255.0

final_image.shape

new_model.predict(final_image)

'''
Realtime Video Capture

Firstly to detect if eyes are closed or open

if eyes are closed for unusual time
'''
frequency = 2500 # Set Frequency to 2500 Hz
duration = 1000 # Set duration to 1000 ms = 1 second
cap = cv2.VideoCapture(0) # Setting default webcam for Video Capture
cap.set(3,600)
cap.set(4,400)
if not cap.isOpened(): # If webcam dosen't open then this raises an error
    raise IOError("Cannot Open Webcam")
counter = 0
while True:
    ret,frame = cap.read()
    
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,5)
    eyes = eye_cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        
    for(x,y,w,h) in eyes:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        eyess = eye_cascade.detectMultiScale(roi_color)
        if len(eyess) != 0:
            for (ex,ey,ew,eh) in eyess:
                eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    final_image = cv2.resize(eyes_roi,(224,224))
    final_image = np.expand_dims(final_image,axis=0) ##need fourth dimension
    final_image = final_image/255.0
    
    Predictions = new_model.predict(final_image)
    if (Predictions>0.05):
        status = "Open Eyes"
        cv2.putText(frame,
                    status,
                    (150,150),
                    font,3,
                    (0,255,0),
                    2,
                    cv2.LINE_4)
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
    
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        # Add text
        cv2.putText(frame,'Active',(x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    else:
        counter = counter + 1
        status = "Closed Eyes"
        cv2.putText(frame,
                    status,
                    (150,150),
                    font,3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x + w,y + h),(0,0,255), 2)
        
        if counter>5:
            x1,y1,w1,h1 = 0,0,175,75 # Draw black background rectangle
            cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
            # Add text
            cv2.putText(frame,'Sleep Alert !!',(x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            winsound.Beep(frequency,duration)
            counter = 0
        
    cv2.imshow("My WebCam",frame)    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

frequency = 2500 # Set Frequency to 2500 Hz
duration = 1000 # Set duration to 1000 ms = 1 second
# from deepface import DeepFace
cap = cv2.VideoCapture(0) # Setting default webcam for Video Capture
cap.set(3,400)
cap.set(4,500)

if not cap.isOpened(): # If webcam dosen't open then this raises an error
    raise IOError("Cannot Open Webcam")
counter = 0

while True:
    ret,frame = cap.read()
    cv2.imshow("camera",frame)
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
    eyes = eye_cascade.detectMultiScale(frame,1.1,5)
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    
    for (x,y,w,h) in eyes:
        #roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        eyess = eye_cascade.detectMultiScale(roi_color)
        if len(eyess) == 0:
            print("eyes are not detected")
        else:
            for (ex,ey,ew,eh) in eyess:
                eyes_roi = roi_color[ey:ey+eh, ex:ex+ew]
    
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #print(faceCascade.empty())
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(frame,1.1,5)
    
    # Draw a rectangle around the faces
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    final_image = cv2.resize(eyes_roi,(224,224))
    final_image = np.expand_dims(final_image,axis=0) ##need fourth dimension
    final_image = final_image/255.0
    
    Predictions = new_model.predict(final_image)
    if (Predictions>0.8):
        status = "Open Eyes"
        cv2.putText(frame,
                    status,
                    (150,150),
                    font,3,
                    (0,255,0),
                    2,
                    cv2.LINE_4)
        x1,y1,w1,h1 = 0,0,175,75
        # Draw black background rectangle
        
        cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
        # Add text
        cv2.putText(frame,'Active',(x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    else:
        counter = counter + 1
        status = "Closed Eyes"
        cv2.putText(frame,
                    status,
                    (150,150),
                    font,3,
                    (0,0,255),
                    2,
                    cv2.LINE_4)
        cv2.rectangle(frame,(x,y),(x + w,y + h),(0,0,255), 2)
        if counter>5:
            x1,y1,w1,h1 = 0,0,175,75 # Draw black background rectangle
            cv2.rectangle(frame,(x1,x1),(x1 + w1,y1 + h1),(0,0,0), -1)
            # Add text
            cv2.putText(frame,'Sleep Alert !!',(x1 + int(w1/10),y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            winsound.Beep(frequency,duration)
            counter = 0
            
    cv2.imshow('Drowsiness Detection',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    cap.release()
    cv2.destroyAllWindows()