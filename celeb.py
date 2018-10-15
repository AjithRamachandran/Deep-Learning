import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.utils import to_categorical, normalize
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random
from tensorflow.keras.utils import normalize

df = pd.read_csv('attr.csv')
df = df.replace(-1, 0)
images = df['image_id']
df = df.loc[0:29999]
labels = df[['Male', 'Young', 'Smiling', 'Eyeglasses']]

def image_to_array(array, type_):
    array = []
    for i in type_:
        img = cv2.imread('img/'+ i)
        array.append(img)
    return array

train = images[0:24999]
test = images[25000:27499]
val = images[27500:29999]
y_train = labels[0:24999]
y_test = labels[25000:27499]
y_val = labels[27500:29999]

X_train = []
X_test = []
X_val = []
X_train = image_to_array(X_train, train)
X_test = image_to_array(X_test, test)
X_val = image_to_array(X_val, val)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val) 

X_train = X_train.reshape(X_train.shape[0], 218, 178, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 218, 178, 1).astype('float32')
X_val = X_val.reshape(X_val.shape[0], 218, 178, 1).astype('float32')

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(218, 178, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.20))
 
model.add(Dense(4))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2, mode='auto')

# model.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[early_stop])

# model.save('celeb.model')

model = load_model('celeb.model')

count = 0
pred = model.predict(X_test)
# for i in range (0,len(X_test)):
    # for j in range(0, 4):
        # pred[i][j] = pred[i][j]*100
        # if(pred[i][j]>75):
            # pred[i][j]=1
        # else:
            # pred[i][j]=0
    # if(np.array_equal(y_test[i],pred[i])):
        # count+=1
# score = (count/len(pred))*100
# print(score)

random = random.randint(0,2499)

if(pred[random][0] == 1):
    gender = 'Male'
else:
    gender = 'Female'
if(pred[random][1] == 1):
    age = 'Young'
else:
    age = 'Old'
if(pred[random][2] == 1):
    smile = 'Smiling'
else:
    smile = 'Not Smiling'
if(pred[random][3] == 1):
    glass = 'Eyeglasses'
else:
    glass = 'No Eyeglasses'

label = gender + '\n' + smile + '\n' + glass

print(label)

output = X_test[random]

cv2.putText(output, label, (10,500), cv2.FONT_HERSHEY_COMPLEX, 4, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow('output', output)
cv2.waitKey()

