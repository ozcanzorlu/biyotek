import tensorflow as tf
from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ---- MODEL ----
interpreter = tf.lite.Interpreter(model_path="skin_cancer_cnn.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dinamik input size alma
input_shape = input_details[0]['shape']   # örn: [1,128,128,3]
IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]

# Upload klasörü
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sınıflar (Türkçe açıklamalar)
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

@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    confidence = None
    risk_score = None
    img_path = None
    family_history = None
    risk_factors = []

    if request.method == 'POST':
        file = request.files['file']
        family_history = request.form.get('family_history')
        risk_factors = request.form.getlist('risk_factors')

        if file and file.filename != '':
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Model tahmini
            result, conf = predict_image(file_path)
            class_codes = list(classes.keys())
            code = class_codes[result]
            prediction = classes[code]
            confidence = round(conf * 100, 2)
            img_path = file_path

            # Risk skoru hesaplama
            risk_score = confidence
            if family_history == "Evet":
                risk_score += 10
            if "uv" in risk_factors:
                risk_score += 5
            if "open_skin" in risk_factors:
                risk_score += 5
            if risk_score > 100:
                risk_score = 100

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           risk_score=risk_score,
                           img_path=img_path,
                           family_history=family_history,
                           risk_factors=risk_factors)

if __name__ == '__main__':
    app.run(debug=True)