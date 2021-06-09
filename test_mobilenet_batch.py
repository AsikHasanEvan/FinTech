import argparse
import pickle
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pathlib
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
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

images_path = pathlib.Path(args["image"])
images_path = list(images_path.glob("*"))

inputs = []
file_names = []

for img_path in images_path:
    file_name = str(img_path).split(os.path.sep)[-1]
    ###
    # Predict
    ##
    img_path = str(img_path)
    input_arr = cv2.imread(img_path)
    input_arr = cv2.cvtColor(input_arr, cv2.COLOR_BGR2RGB)
    input_arr = crop_face(input_arr)

    input_arr = cv2.resize(input_arr, (224, 224))

    print(input_arr.shape)

    inputs.append(input_arr)
    file_names.append(file_name)

predictions = mobilenetv2.predict(np.array(inputs))

predictions = np.argmax(predictions, axis=1)

print(predictions)

i=0
true_pred = 0
for l in file_names:
    result = "fake" if labels["fake"] == predictions[i] else "real"
    print(f"Actual: {l} and Predicted: {result}")
    label_num = 1 if l.find("real") != -1 else 0
    if label_num == predictions[i]:
        true_pred += 1
    i+=1

print(f"Total Prediction: {i} , Correct Preds: {true_pred}, Incorrect Preds: {i - true_pred}")

    
