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

def predict_image(img_path):
    img = Image.open(img_path).resize((64, 64))  # Model giriş boyutu
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(prediction)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Dosya bulunamadı!"
    file = request.files['file']
    if file.filename == '':
        return "Dosya seçilmedi!"

    # Klasör yoksa oluştur
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    result = predict_image(file_path)

    # Tahmin sınıfları – bunları kendi veri setine göre düzenle
    classes = ["Benign", "Kötü huylu (Melanom)"]

    return render_template('result.html', prediction=classes[result])

if __name__ == "__main__":
    app.run(debug=True)