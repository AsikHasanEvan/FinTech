import argparse
import pickle
import cv2
import os
import pathlib
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model




def crop_face(img):
    face_detector = MTCNN()
    face = face_detector.detect_faces(img)
    if len(face) > 0:
        box = face[0]['box']
        # print("Box: ", box)
        startX = box[0]
        startY = box[1]
        endX = startX + box[2]
        endY = startY + box[3]
        roi_img_array = img[startY: endY, startX: endX]
        return roi_img_array
    else:
        return None


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-l", "--label", type=str, required=True,
                help="path to labels")
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to image")
args = vars(ap.parse_args())


###
# Importing keras model
###

modelPath = args["model"]
mobilenetv2 = load_model(modelPath)
labels = pickle.loads(open(args["label"], "rb").read())
print("Labels: ", labels)

image_path = str(pathlib.Path(args["image"]))

###
# Predict
###

# image = tf.keras.preprocessing.image.load_img(image_path)
# input_arr = tf.keras.preprocessing.image.img_to_array(image)

input_arr = cv2.imread(image_path)
input_arr = cv2.cvtColor(input_arr, cv2.COLOR_BGR2RGB)
input_arr = crop_face(input_arr)

input_arr = cv2.resize(input_arr, (224, 224))

cv2.imshow("img", cv2.cvtColor(input_arr, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)

input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = mobilenetv2.predict(input_arr)

print(predictions)

result = "fake" if labels["fake"] == np.argmax(predictions) else "real"

print("Final Result: ", result)