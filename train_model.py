import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
# ===================
# 1. Metadata oku
# ===================
metadata = pd.read_csv("data/HAM10000_metadata.csv")

# ===================
# 2. Görselleri yükle
# ===================
IMG_SIZE = 128
image_dir1 = "data/HAM10000_images_part_1/"
image_dir2 = "data/HAM10000_images_part_2/"

images = []
labels = []

for idx, row in metadata.iterrows():
    file_name = row['image_id'] + ".jpg"
    if os.path.exists(os.path.join(image_dir1, file_name)):
        img_path = os.path.join(image_dir1, file_name)
    else:
        img_path = os.path.join(image_dir2, file_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    images.append(img)
    labels.append(row['dx'])

X = np.array(images) / 255.0
y = np.array(labels)

# ===================
# 3. Etiketleri dönüştür
# ===================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ===================
# 4. Eğitim/test ayır
# ===================
X_train, X_val, y_train, y_val = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# ===================
# 5. CNN Modeli
# ===================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(y_categorical.shape[1], activation='softmax')  # sınıf sayısı kadar çıktı
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ===================
# 6. Modeli eğit
# ===================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# ===================
# 7. Kaydet
# ===================
model.save("skin_cancer_cnn.h5")
print("✅ Model kaydedildi: skin_cancer_cnn.h5")

val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=2)
print(f"📊 Doğrulama Loss: {val_loss:.4f}")
print(f"✅ Doğrulama Accuracy: {val_accuracy:.4f}")


# Loss grafiği
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.legend()
plt.title("Loss Eğrisi")
plt.show()

# Accuracy grafiği
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')
plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')
plt.legend()
plt.title("Accuracy Eğrisi")
plt.show()