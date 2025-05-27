import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. Charger le modèle entraîné
model = load_model("line_segment_model.h5")

# 2. Charger une image de test
def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32) / 255.0  # Normalisation
    img = tf.expand_dims(img, axis=0)  # Ajouter batch dimension
    return img

image_path = "dataset/CapturedImages/image_0_image_1000.png"
input_image = load_and_preprocess_image(image_path)

# 3. Prédire le masque
pred_mask = model.predict(input_image)[0]  # Enlever la batch dimension
pred_mask = tf.where(pred_mask > 0.5, 1, 0)  # Binarisation

# 4. Afficher le résultat
plt.figure(figsize=(10, 5))

# Image d'origine
plt.subplot(1, 2, 1)
plt.title("Image d'entrée")
plt.imshow(tf.squeeze(input_image))

# Masque prédit
plt.subplot(1, 2, 2)
plt.title("Masque prédit")
plt.imshow(tf.squeeze(pred_mask), cmap='gray')

plt.show()
