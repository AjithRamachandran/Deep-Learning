import os
import numpy as np
from scipy.misc import imsave
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation, Dropout, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

path = 'data/'
X = []
y = []

def create_data():
    for img in os.listdir(path):
        try:
            image = load_img(path + img, target_size=(256, 256))
            image = img_to_array(image)
            data = rgb2lab(1.0/255*image)[:,:,0]
            label = rgb2lab(1.0/255*image)[:,:,1:]
            label = label/128
            X.append(data)
            y.append(label)
        except Exception as e:
            print(e)

create_data()

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], 256, 256, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(256, 256, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(2, (3, 3), activation='tanh'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=1, mode='auto')
model.fit(X_train, y_train, batch_size=1, epochs=1, validation_data=(X_val, y_val), callbacks=[early_stop])

model.save('color.model')

# model = load_model('color.model')

test = load_img('/home/unicorn/color/test.jpg', target_size=(256, 256))
test = img_to_array(test)
test = rgb2lab(1.0/255*test)[:,:,0]
test = test.reshape(1, 104, 104, 1)

output = model.predict(test)
output = output * 128

cur = np.zeros((256, 256, 3))
cur[:,:,0] = test[:,:,0]
cur[:,:,1:] = output
imsave("result.jpg", lab2rgb(cur))