import tensorflow

import pandas as pd
import numpy as np
import os
import cv2
import math
import pathlib

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Convolution2D, BatchNormalization
from tensorflow.keras.layers import Flatten, MaxPooling2D, Dropout

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from preprocess_image import custom_preprocess

print("Tensorflow-version:", tensorflow.__version__)

### Model Fine Tuning ###

base_model = DenseNet121(
    weights='imagenet', include_top=True, input_shape=(224, 224, 3))

x = base_model.output

preds = Dense(2, activation='softmax')(x)  # FC-layer

model = Model(inputs=base_model.input, outputs=preds)
model.summary()

### Freezing Base model weights for avoiding overfitting ###
# for layer in model.layers[:-5]:
#     layer.trainable = False

# for layer in model.layers[-5:]:
#     layer.trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


### Prepare dataset ###
data_path = "dataset/validation"
data_dir = pathlib.Path(data_path)

images_path = list(data_dir.glob("*/*"))

data = []
labels = []

counter = 0
for img in images_path:
    label = str(img).split(os.path.sep)[-2]
    image = cv2.imread(str(img))
    image = custom_preprocess(image)

    data.append(image)
    labels.append(label)

    print(counter)
    counter += 1

data = np.array(data, dtype="float32")
labels = np.array(labels)

binarizer = LabelBinarizer()

labels = binarizer.fit_transform(labels)

print("Class Index: ", binarizer.classes_)

X_train, X_test, Y_train, Y_test = train_test_split(
    data, labels, test_size=0.3, random_state=42)

# print(X_train.shape)
# print(X_test.shape)

### fit model ###

anne = ReduceLROnPlateau(monitor='val_accuracy',
                         factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint(
    'densenet_model.h5', verbose=1, save_best_only=True)

datagen = ImageDataGenerator(
    zoom_range=0.2, horizontal_flip=True, shear_range=0.2)

datagen.fit(X_train)

# Fits-the-model
history = model.fit(x=datagen.flow(X_train, Y_train, batch_size=64),
                    steps_per_epoch=X_train.shape[0] // 64,
                    epochs=5,
                    verbose=1,
                    callbacks=[anne, checkpoint],
                    validation_data=datagen.flow(
                        X_train, Y_train, batch_size=64),
                    validation_steps=X_train.shape[0] // 64
                    )


### Accuracy visualization ###
Y_pred = model.predict(X_test)

total = 0
accurate = 0
accurateindex = []
wrongindex = []

for i in range(len(Y_pred)):
    if np.argmax(Y_pred[i]) == np.argmax(Y_test[i]):
        accurate += 1
        accurateindex.append(i)
    else:
        wrongindex.append(i)

    total += 1

print('Total-test-data;', total, '\taccurately-predicted-data:',
      accurate, '\t wrongly-predicted-data: ', total - accurate)
print('Accuracy:', round(accurate/total*100, 3), '%')


