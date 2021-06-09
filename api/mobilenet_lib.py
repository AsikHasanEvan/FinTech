import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
try:
    tf_gpus = tf.config.list_physical_devices('GPU')
    for gpu in tf_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import numpy as np
import pathlib
import pickle
import cv2
import base64
from PIL import Image
from io import BytesIO





modelPath = str(pathlib.Path("../artifacts/liveness_model.h5"))
mobilenetv2 = load_model(modelPath)

label_path = str(pathlib.Path("../artifacts/label.pickle"))
labels = pickle.loads(open(label_path, "rb").read())

face_detector = MTCNN()


def base64_to_numpy(base64_image):
    decoded = base64.b64decode(base64_image)
    img = np.array(Image.open(BytesIO(decoded)))
    return img

def crop_face(img):
    
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


def detect_liveness(img_path):
    img_path = str(pathlib.Path(img_path))
    input_arr = cv2.imread(img_path)
    input_arr = cv2.cvtColor(input_arr, cv2.COLOR_BGR2RGB)
    input_arr = crop_face(input_arr)

    input_arr = cv2.resize(input_arr, (224, 224))

    # cv2.imshow("img", cv2.cvtColor(input_arr, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = mobilenetv2.predict(input_arr)

    print(predictions)

    result = "fake" if labels["fake"] == np.argmax(predictions) else "real"

    return result


def detect_liveness_base64(img_base64):
    
    input_arr = base64_to_numpy(img_base64)

    input_arr = crop_face(input_arr)

    input_arr = cv2.resize(input_arr, (224, 224))

    # cv2.imshow("img", cv2.cvtColor(input_arr, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = mobilenetv2.predict(input_arr)

    print(predictions)

    result = "fake" if labels["fake"] == np.argmax(predictions) else "real"

    return result