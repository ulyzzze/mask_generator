import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import matplotlib.pyplot as plt
import numpy as np

# Définir l'augmentation
augmentation_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.6),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
    A.RandomShadow(p=0.2),
    A.Resize(height=224, width=224)  # Taille cible du modèle
], additional_targets={"mask": "mask"})

# Charger image et masque avec vérification des chemins
image_path = '../dataset/CapturedImages/image_0_image_1.png'  # Ajustez le chemin relatif
mask_path = '../dataset/mask/image_0_mask_1.png'  # Ajustez le chemin relatif

# Vérifier que les fichiers existent
if not os.path.exists(image_path):
    print(f"Erreur: Image non trouvée à {image_path}")
    exit(1)
    
if not os.path.exists(mask_path):
    print(f"Erreur: Masque non trouvé à {mask_path}")
    exit(1)

# Charger image et masque
image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Vérifier que les images ont été chargées
if image is None:
    print(f"Erreur: Impossible de charger l'image {image_path}")
    exit(1)
    
if mask is None:
    print(f"Erreur: Impossible de charger le masque {mask_path}")
    exit(1)

print(f"Image chargée: {image.shape}")
print(f"Masque chargé: {mask.shape}")

# Convertir BGR vers RGB pour matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for i in range(5503):
    img = cv2.imread(f'dataset/CapturedImages/img_0_image_{i}.png')
    mask = cv2.imread(f'dataset/mask/mask_0_mask_{i}.png', cv2.IMREAD_GRAYSCALE)

    for j in range(2):  # deux augmentations par image
        augmented = augmentation_pipeline(image=img, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        cv2.imwrite(f'augmented/images/img_{i}_{j}.png', aug_img)
        cv2.imwrite(f'augmented/masks/mask_{i}_{j}.png', aug_mask)
        # Convertir l'image augmentée BGR vers RGB
        image_aug_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)

        print("Augmentation appliquée avec succès!")
        print(f"Image augmentée: {aug_img.shape}")
        print(f"Masque augmenté: {aug_mask.shape}")


# Afficher les images
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Image originale
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Image Originale')
axes[0, 0].axis('off')

# Masque original
axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title('Masque Original')
axes[0, 1].axis('off')

# Image augmentée
axes[1, 0].imshow(image_aug_rgb)
axes[1, 0].set_title('Image Augmentée')
axes[1, 0].axis('off')

# Masque augmenté
axes[1, 1].imshow(mask_aug, cmap='gray')
axes[1, 1].set_title('Masque Augmenté')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()