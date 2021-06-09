# Using Tensorflow-2.4.x
import tensorflow as tf
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# set the matplotlib backend so figures can be saved in the background
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, MobileNet
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Activation, Flatten
# from imutils import paths
import numpy as np
import cv2
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


###
# Reading and parsing the arguments
###
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-a", "--artifacts", type=str, required=True,
                help="path to artifacts")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())


###
# Global constant variable init
###

INIT_LR = 0.0001
BS = 32
EPOCHS = 10

train_data_path = args["dataset"] + "/train"
validation_data_path = args["dataset"] + "/validation"


###
# Model Creation from mobilenet
###
print("[INFO] creating model...")
mobilenetv2 = MobileNetV2(
    weights='imagenet', input_shape=(224, 224, 3))

mobilenetv2_output = mobilenetv2.output
preds = Dense(2, activation='softmax')(mobilenetv2_output)

custom_mobilenetv2 = Model(inputs=mobilenetv2.input, outputs=preds)

###
# Data preprocessing for mobilenetv2
###

train_batches = ImageDataGenerator().flow_from_directory(
    train_data_path, target_size=(224, 224), batch_size=BS
)

validation_batches = ImageDataGenerator().flow_from_directory(
    validation_data_path, target_size=(224, 224), batch_size=BS
)


# print(train_batches.class_indices)

custom_mobilenetv2.compile(
    Adam(INIT_LR), loss="categorical_crossentropy", metrics=['accuracy'])

step_size_train = train_batches.n//train_batches.batch_size
step_size_val = validation_batches.n//validation_batches.batch_size

H = custom_mobilenetv2.fit(x=train_batches,
                           steps_per_epoch=step_size_train,
                           epochs=EPOCHS,
                           validation_data=validation_batches,
                           validation_steps=step_size_val,
                           verbose=1,
                           use_multiprocessing=False)


###
# Saving the model and label
###


# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["artifacts"]))
model_path = os.path.abspath(args["artifacts"] + "/liveness_model.h5")
label_path = os.path.abspath(args["artifacts"] + "/label.pickle")

custom_mobilenetv2.save(model_path)

f = open(label_path, "wb")
f.write(pickle.dumps(train_batches.class_indices))
f.close()


# plot the training loss and accuracy
print("[INFO] plotting the result...")
print(H.history)
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])