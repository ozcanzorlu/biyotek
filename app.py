import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_cancer_cnn.tflite")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

# Model yükleme
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model başarıyla yüklendi.")
except Exception as e:
    print("❌ Model yüklenemedi:", e)

# Türkçe hastalık sınıfları
classes = {
    0: {'name': 'akiec', 'desc': 'Aktinik Keratoz / Bowen Hastalığı (Orta Risk)', 'severity': 'orta'},
    1: {'name': 'bcc', 'desc': 'Bazal Hücreli Karsinom (Yüksek Risk)', 'severity': 'yüksek'},
    2: {'name': 'bkl', 'desc': 'Benign Keratoz (Zararsız Deri Lezyonu)', 'severity': 'düşük'},
    3: {'name': 'df', 'desc': 'Dermatofibroma (Zararsız Deri Nodülü)', 'severity': 'düşük'},
    4: {'name': 'mel', 'desc': 'Melanom (Kritik Risk)', 'severity': 'kritik'},
    5: {'name': 'nv', 'desc': 'Melanositik Nevüs (Ben, Genellikle Zararsız)', 'severity': 'düşük'},
    6: {'name': 'vasc', 'desc': 'Vasküler Lezyon (Damar Kayması)', 'severity': 'düşük'}
}

def preprocess_image(image):
    img = image.resize((128, 128))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def calculate_age_factor(age):
    if age < 30:
        return 0.1
    elif age < 50:
        return 0.3
    elif age < 70:
        return 0.6
    else:
        return 1.0

def calculate_skin_type_factor(skin_type):
    skin_type = int(skin_type)
    if skin_type <= 2:
        return 1.0
    elif skin_type <= 4:
        return 0.6
    else:
        return 0.2

def calculate_risk_score(confidence, age, skin_type, family_history, additional_risks):
    base_score = confidence
    age_factor = calculate_age_factor(age)
    skin_factor = calculate_skin_type_factor(skin_type)
    family_factor = 1.0 if family_history == "yes" else 0.0

    extra_factor = 0.0
    if additional_risks:
        if "sun_exposure" in additional_risks:
            extra_factor += 0.4
        if "outdoor_work" in additional_risks:
            extra_factor += 0.4
        if "solarium" in additional_risks:
            extra_factor += 0.3

    personal_factor = min(age_factor + skin_factor + family_factor + extra_factor, 3.0) / 3.0
    risk_score = (base_score * 0.7) + (personal_factor * 30)
    return min(risk_score, 100)

def get_risk_category(risk_score, disease_name, severity):
    # Melanomda her zaman acil uyarı
    if "Melanom" in disease_name:
        return {
            "category": "KRİTİK",
            "color": "danger",
            "advice": "Melanom tespit edildi! Acil dermatoloğa başvurunuz."
        }

    # Zararsız hastalıklar için öneri
    if severity == 'düşük':
        return {
            "category": "ZARARSIZ",
            "color": "success",
            "advice": "Bu lezyon genellikle zararsızdır. Düzenli takip önerilir."
        }

    # Diğer hastalıklar için risk skoruna göre öneri
    if risk_score < 40:
        return {
            "category": "DÜŞÜK",
            "color": "success",
            "advice": "Genellikle zararsızdır. Takip önerilir."
        }
    elif risk_score < 70:
        return {
            "category": "ORTA",
            "color": "warning",
            "advice": "Kontrol gerekebilir. Dermatoloğa başvurmanız faydalı olabilir."
        }
    else:
        return {
            "category": "YÜKSEK",
            "color": "danger",
            "advice": "Yüksek risk! Acil doktora başvurunuz."
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Lütfen bir fotoğraf yükleyin.")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Lütfen bir fotoğraf seçin.")

        try:
            age = int(request.form.get('age', 30))
            skin_type = request.form.get('skin_type', '3')
            family_history = request.form.get('family_history', 'no')
            additional_risks = request.form.getlist('additional_risks')

            image = Image.open(file.stream)
            input_data = preprocess_image(image)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            pred_index = np.argmax(output_data)
            confidence = float(np.max(output_data)) * 100

            disease_info = classes[pred_index]
            risk_score = calculate_risk_score(confidence, age, skin_type, family_history, additional_risks)
            risk_info = get_risk_category(risk_score, disease_info['desc'], disease_info['severity'])

            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()

            return render_template('index.html',
                                   prediction=disease_info['desc'],
                                   confidence=round(confidence, 2),
                                   risk_score=round(risk_score, 2),
                                   risk_category=risk_info['category'],
                                   risk_color=risk_info['color'],
                                   advice=risk_info['advice'],
                                   image_data=img_str,
                                   age=age,
                                   skin_type=skin_type,
                                   family_history=family_history,
                                   additional_risks=additional_risks)
        except Exception as e:
            return render_template('index.html', error=f"Hata oluştu: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)