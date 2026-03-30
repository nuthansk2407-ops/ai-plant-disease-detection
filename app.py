import os
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import gdown  # better for Google Drive

# 🔗 Put your file ID here (important)
FILE_ID = "1Z6kMwiEt5vPQfnKcw7OmPPOkBhlTKK1O"
MODEL_PATH = "model/plant_model.h5"

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📥 Download model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    print("Downloading model...")
    
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
