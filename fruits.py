import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv2D, Activation, Dropout, BatchNormalization, MaxPooling2D, Flatten

train_path = 'Train'
test_path = 'Test'
categories = []
IMG_SIZE = 100
X = []
y = []
X_test = []
y_test = []

for item in os.listdir(train_path):
    categories.append(item)

def create_data(path_):
    data = []
    for category in categories:
        path = os.path.join(path_, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img))
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append([image, class_num])
            except Exception as e:
                print(e)
    return data

def train_test(train, test):
    for features,label in train:
        X.append(features)
        y.append(label)

    for features,label in test:
        X_test.append(features)
        y_test.append(label)

def create_model():
    model = Sequential()

    model.add(Conv2D(16, (5, 5), input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.05))

    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(69))
    model.add(Activation('softmax'))

    return model

train_data = create_data(train_path)
test_data = create_data(test_path)
random.shuffle(train_data)
random.shuffle(test_data)

train_test(train_data, test_data)

y = to_categorical(y)
X = np.array(X)
X_test = np.array(X_test)
X = X/255.0
X_test = X_test/255.0

X_train = X[0:26999]
y_train = y[0:26999]
X_val = X[27000:]
y_val = y[27000:]

X_train = X_train.reshape(X_train.shape[0], 100, 100, 3)
X_test = X_test.reshape(X_test.shape[0], 100, 100, 3)

model = create_model()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=1, mode='auto')
model.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val))
model.save('fruit.model')

# model = load_model('fruit_sigmoid.model')
# model = load_model('fruit_softmax.model')

count = 0
prediction = model.predict(X_test)
for i in range(0, len(y_test)):
    if(np.argmax(prediction[i]) == y_test[i]):
        count=count+1
score = (count/len(X_test))*100
print(score)

test = cv2.imread('test.jpg')
test = cv2.resize(test, (100, 100))
test = test.reshape(1, 100, 100, 3)
test = np.array(test)

prediction = model.predict(test)
print(prediction)