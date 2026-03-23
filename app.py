import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model("model/plant_model.h5")

with open("model/classes.txt") as f:
    class_names = [line.strip() for line in f]

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))
    return img

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)

    prediction = model.predict(img)
    index = np.argmax(prediction)

    result = class_names[index]
    confidence = round(np.max(prediction) * 100, 2)

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           image_path=filepath)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)