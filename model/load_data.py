import cv2
import os
import numpy as np

def load_data(image_dir, mask_dir, size=(256, 256)):
    images = []
    masks = []

    files = os.listdir(image_dir)

    image_files_raw = [f for f in files if "_image_" in f]

    def get_sort_key(filename):
        try:
            # Sépare le nom de fichier par "_" et prend l'avant-dernier élément (le numéro)
            # puis enlève l'extension ".png"
            # Exemple: "image_0_image_5479.png" -> "5479"
            numeric_part = filename.split('_')[-1].split('.')[0]
            return int(numeric_part)
        except (IndexError, ValueError):
            # En cas d'erreur (format de nom de fichier inattendu),
            # retourner une valeur qui le placera à la fin ou au début
            return -1 # ou float('inf')
        
    image_files = sorted(image_files_raw, key=get_sort_key)

    for img_file in image_files:
        # Crée le nom du masque correspondant
        mask_file = img_file.replace("_image_", "_mask_")

        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Masque manquant pour {img_file}, ignoré.")
            continue

        # Chargement des images
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Redimensionnement
        img = cv2.resize(img, size)
        mask = cv2.resize(mask, size)

        # Normalisation
        img = img / 255.0
        mask = (mask > 127).astype(np.uint8)  # Binarisation

        images.append(img)
        masks.append(mask)

    X = np.array(images)
    Y = np.array(masks)[..., np.newaxis]  # Pour avoir la forme (batch, h, w, 1)

    return X, Y