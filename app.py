import tensorflow as tf
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# TFLite modeli yükle
interpreter = tf.lite.Interpreter(model_path="skin_cancer_cnn.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def predict_image(img_path):
    img = Image.open(img_path).resize((128, 128))  # Model giriş boyutu 128x128
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(prediction), float(np.max(prediction))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="Dosya bulunamadı!")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="Dosya seçilmedi!")

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        result, conf = predict_image(file_path)

        classes = ["Benign", "Kötü huylu (Melanom)"]  # kendi sınıflarını buraya yaz
        prediction = classes[result]
        confidence = round(conf * 100, 2)
        img_path = file_path

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)