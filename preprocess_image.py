import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array


def custom_preprocess(img_bgr):
    # img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_bgr, (224, 224))
    return np.array(img, dtype="float32") / 255