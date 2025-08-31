import tensorflow as tf
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# TFLite modelini yükle
interpreter = tf.lite.Interpreter(model_path="skin_cancer_cnn.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Modele göre giriş boyutu
input_shape = input_details[0]['shape']
IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 7 sınıfın Türkçe karşılıkları ve açıklamaları
classes = {
    'akiec': "Aktinik keratoz / Bowen hastalığı - Ciltte güneşe bağlı yüzeysel tümör",
    'bcc': "Bazal Hücreli Karsinom - En sık görülen, genellikle yavaş ilerleyen cilt kanseri",
    'bkl': "Benign Keratoz - İyi huylu deri lezyonu",
    'df': "Dermatofibroma - Zararsız, küçük ve sert deri nodülü",
    'mel': "Melanom - Tehlikeli ve hızlı yayılabilen cilt kanseri",
    'nv': "Melanositik Nevüs (Ben) - Çoğunlukla iyi huylu benler",
    'vasc': "Vasküler Lezyon - Damar kaynaklı lezyon (ör: hemanjiyom)"
}

def predict_image(img_path):
    img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
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
        file = request.files['file']
        if file and file.filename != '':
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            result, conf = predict_image(file_path)

            # index → class kodu
            class_codes = list(classes.keys())
            if result < len(class_codes):
                code = class_codes[result]
                prediction = classes[code]   # Türkçe açıklamalı
                confidence = round(conf * 100, 2)
                img_path = file_path
            else:
                prediction = f"Geçersiz sınıf indexi: {result}"

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)