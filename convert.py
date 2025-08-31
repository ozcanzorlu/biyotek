import tensorflow as tf
from tensorflow.keras.models import load_model

# .h5 dosyasını model olarak yükle
model = load_model("skin_cancer_cnn.h5")

# Keras modelini TFLite'a dönüştür
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# .tflite olarak kaydet
with open("skin_cancer_cnn.tflite", "wb") as f:
    f.write(tflite_model)

print("Dönüştürme tamam ✅ -> skin_cancer_cnn.tflite")