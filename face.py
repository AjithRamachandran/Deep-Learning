import numpy as np
import cv2
import os
import random
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

CTG = ['al gore', 'bill gates', 'michelle obama', 'steve jobs']
IMG_SIZE = 100
DIR = 'data'

training_data = []

def create_training_data():
    for category in CTG:

        path = os.path.join(DIR,category)
        class_num = CTG.index(category)
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                training_data.append([image, class_num])
            except Exception as e:
                pass
create_training_data()
random.shuffle(training_data)
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

y = to_categorical(y)
X = np.array(X)
X = X/255.0
X = X.reshape(X.shape[0], 100, 100, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

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
early_stop = EarlyStopping(monitor='val_loss', patience=100, mode='auto')
model.fit(X_train, y_train, batch_size=32, epochs=5000, validation_data=(X_val, y_val))
model.save('face.model')

test = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
test = cv2.resize(test, (IMG_SIZE, IMG_SIZE))
test = test.reshape(1, 100, 100, 1)

pred = model.predict(test)

print(pred)

