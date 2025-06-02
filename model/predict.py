import cv2
import numpy as np
from model import get_compiled_unet
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
model = load_model("model.h5")  # ou get_compiled_unet() si tu n'as pas sauvegardé encore

# Charger une image à prédire
img_path = "dataset/CapturedImages/image_0_image_42.png"  # ← modifie selon ton cas
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (256, 256))
img_input = img_resized / 255.0
img_input = np.expand_dims(img_input, axis=0)  # forme (1, 256, 256, 3)

# Prédiction
pred = model.predict(img_input)
mask_pred = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255  # seuillage binaire

# Affichage avec OpenCV
cv2.imshow("Image originale", img_resized)
cv2.imshow("Masque prédit", mask_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()
