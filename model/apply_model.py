import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import save_img

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

image_path = "dataset/CapturedImages/image_0_image_1.png"
input_image = load_and_preprocess_image(image_path)

# 3. Prédire le masque
pred_mask_raw = model.predict(input_image)[0]

binary_mask = tf.where(pred_mask_raw > 0.5, 1.0, 0.0) # Assurer que les valeurs sont float
save_img("predicted_mask.png", binary_mask)

# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.title("Image d'entrée")
# plt.imshow(tf.squeeze(input_image))

# plt.subplot(1, 3, 2)
# plt.title("Masque Prédit (Brut)")
# plt.imshow(tf.squeeze(pred_mask_raw), cmap='gray', vmin=0, vmax=1) # Afficher les niveaux de gris
# plt.colorbar()

# plt.subplot(1, 3, 3)
# plt.title("Masque Prédit (Seuil 0.5)")
# plt.imshow(tf.squeeze(tf.where(pred_mask_raw > 0.5, 1, 0)), cmap='gray')

# plt.show()
