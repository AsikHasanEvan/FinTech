import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess_image import custom_preprocess

model = load_model("densenet_model.h5")
image = cv2.imread("images/4947.png")
print(image.shape)
image_processed = custom_preprocess(image)
image_processed = np.expand_dims(image_processed, axis=0)
print(image_processed.shape)

preds = model.predict(image_processed)

print(preds)
result = np.argmax(preds)
print("fake" if result == 0 else "real")

cv2.imshow("image", cv2.resize(image, (240, 240)))
cv2.waitKey(0)