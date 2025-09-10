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

# Cilt tipi açıklamaları
skin_types = {
    '1': 'Tip I - Çok açık ten, kolay yanar',
    '2': 'Tip II - Açık ten, genelde yanar',
    '3': 'Tip III - Buğday ten, orta derecede yanar',
    '4': 'Tip IV - Esmer ten, nadiren yanar',
    '5': 'Tip V - Koyu esmer, nadiren yanar',
    '6': 'Tip VI - Çok koyu ten, neredeyse hiç yanmaz'
}

# --- Risk hesaplama yardımcı fonksiyonlar ---
def calculate_age_factor(age):
    if age < 30: return 5
    elif age < 50: return 10
    elif age < 70: return 15
    else: return 20

def calculate_skin_type_factor(skin_type):
    skin_type = int(skin_type)
    if skin_type <= 2: return 15
    elif skin_type <= 4: return 10
    else: return 5

def get_risk_category(risk_score):
    if risk_score < 40:
        return {'category': 'DÜŞÜK', 'color': 'success', 'advice': 'Düzenli kontrol yeterli (6 ayda bir önerilir).'}
    elif risk_score < 70:
        return {'category': 'ORTA', 'color': 'warning', 'advice': 'Dikkatli takip gerekli (3 ayda bir önerilir).'}
    else:
        return {'category': 'YÜKSEK', 'color': 'danger', 'advice': 'Acil doktora başvurunuz!'}

def predict_image(img_path):
    img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(prediction), float(np.max(prediction))

# --- FLASK ROUTE ---
@app.route('/', methods=['GET','POST'])
def index():
    prediction = None
    confidence = None
    risk_score = None
    risk_info = None
    img_path = None
    user_info = {}
    risk_factors = []   # ✅ Hata engeli: GET isteğinde varsayılan boş liste

    if request.method == 'POST':
        file = request.files['file']

        # Form verileri
        age = int(request.form.get('age', 25))
        skin_type = request.form.get('skin_type', '3')
        family_history = request.form.get('family_history', 'no')
        risk_factors = request.form.getlist('risk_factors')

        user_info = {
            'age': age,
            'skin_type': skin_types.get(skin_type, 'Bilinmiyor'),
            'family_history': 'Evet' if family_history == 'yes' else 'Hayır'
        }

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

            # Gelişmiş risk skoru
            risk_score = confidence
            risk_score += calculate_age_factor(age)
            risk_score += calculate_skin_type_factor(skin_type)
            if family_history == "yes": risk_score += 15
            if "uv" in risk_factors: risk_score += 5
            if "open_skin" in risk_factors: risk_score += 5

            risk_score = min(risk_score, 100)  # max 100 sınırı
            risk_info = get_risk_category(risk_score)

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           risk_score=risk_score,
                           risk_info=risk_info,
                           img_path=img_path,
                           user_info=user_info,
                           risk_factors=risk_factors)

if __name__ == '__main__':
    app.run(debug=True)