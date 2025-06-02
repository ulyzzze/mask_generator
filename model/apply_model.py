import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from model import dice_coefficient, combined_loss
import os

# 1. Charger le modèle entraîné avec les métriques personnalisées
model = load_model("best_line_segment_model.h5", 
                   custom_objects={
                       'dice_coefficient': dice_coefficient,
                       'combined_loss': combined_loss
                   })

# 2. Fonction de préprocessing améliorée
def load_and_preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32) / 255.0  # Normalisation
    img = tf.expand_dims(img, axis=0)  # Ajouter batch dimension
    return img

# 3. Post-processing amélioré du masque
def postprocess_mask(pred_mask, threshold=0.5):
    """Post-traitement du masque prédit"""
    # Binarisation avec seuil adaptatif
    binary_mask = tf.where(pred_mask > threshold, 1.0, 0.0)
    
    # Optionnel: appliquer des opérations morphologiques
    # (nécessiterait OpenCV ou scipy)
    
    return binary_mask

# 4. Fonction de test sur plusieurs images
def test_multiple_images(image_dir, num_images=5):
    """Tester le modèle sur plusieurs images"""
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files)[:num_images]
    
    fig, axes = plt.subplots(len(image_files), 3, figsize=(15, 5*len(image_files)))
    if len(image_files) == 1:
        axes = axes.reshape(1, -1)
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        
        # Charger et préprocesser
        input_image = load_and_preprocess_image(image_path)
        
        # Prédire
        pred_mask = model.predict(input_image, verbose=0)[0]
        
        # Post-processing
        processed_mask = postprocess_mask(pred_mask, threshold=0.5)
        
        # Affichage
        axes[i, 0].imshow(tf.squeeze(input_image))
        axes[i, 0].set_title(f"Image d'entrée - {filename}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(tf.squeeze(pred_mask), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Masque brut (probabilités)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(tf.squeeze(processed_mask), cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title("Masque final (binarisé)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# 5. Test sur une image spécifique
if __name__ == "__main__":
    # Test sur l'image spécifiée
    image_path = "dataset/CapturedImages/image_0_image_1000.png"
    
    if os.path.exists(image_path):
        input_image = load_and_preprocess_image(image_path)
        
        # Prédiction
        pred_mask = model.predict(input_image)[0]
        processed_mask = postprocess_mask(pred_mask, threshold=0.3)  # Seuil plus bas pour plus de sensibilité
        
        # Affichage
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Image d'entrée")
        plt.imshow(tf.squeeze(input_image))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Masque brut (probabilités)")
        plt.imshow(tf.squeeze(pred_mask), cmap='hot', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Masque final (binarisé)")
        plt.imshow(tf.squeeze(processed_mask), cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('single_test_result.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Statistiques du masque
        total_pixels = tf.size(processed_mask).numpy()
        white_pixels = tf.reduce_sum(processed_mask).numpy()
        percentage = (white_pixels / total_pixels) * 100
        
        print(f"\n=== Statistiques du masque ===")
        print(f"Pixels blancs détectés: {white_pixels:.0f}")
        print(f"Total pixels: {total_pixels}")
        print(f"Pourcentage de lignes: {percentage:.2f}%")
        
    else:
        print(f"Image non trouvée: {image_path}")
        print("Test sur toutes les images du dossier:")
        test_multiple_images("dataset/CapturedImages", num_images=3)