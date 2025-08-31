from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# TFLite modeli yükle
interpreter = tf.lite.Interpreter(model_path="skin_cancer_cnn.tflite")
interpreter.allocate_tensors()

# Girdi ve çıktı detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Görüntüyü modele uygun hale getir
            img = Image.open(img_path).resize((128, 128))
            img_array = np.array(img).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # TFLite model ile tahmin
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])

            prediction = class_names[np.argmax(preds)]
            confidence = round(np.max(preds) * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)