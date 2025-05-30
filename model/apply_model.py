import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import glob
from datetime import datetime

print("🔍 APPLICATION DU MODÈLE OPTIMISÉ")
print("="*50)

# ===== CONFIGURATION =====
# Chemins des modèles (cherche le plus récent)
model_patterns = [
    "line_segment_model_optimized_*.h5",
    "checkpoints/best_model_*.h5", 
    "line_segment_model.h5"
]

model_path = None
for pattern in model_patterns:
    models = glob.glob(pattern)
    if models:
        model_path = max(models, key=os.path.getctime)  # Le plus récent
        break

if not model_path:
    print("❌ Aucun modèle trouvé!")
    print("Modèles recherchés:")
    for pattern in model_patterns:
        print(f"   - {pattern}")
    exit(1)

print(f"📁 Modèle trouvé: {model_path}")

# ===== CHARGEMENT DU MODÈLE =====
print("🔄 Chargement du modèle...")
try:
    model = load_model(model_path)
    print("✅ Modèle chargé avec succès")
    print(f"📐 Architecture: {model.input_shape} -> {model.output_shape}")
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    exit(1)

# ===== FONCTIONS UTILITAIRES =====
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Charge et préprocesse une image pour la prédiction"""
    try:
        # Charger l'image
        img = tf.io.read_file(image_path)
        
        # Décodage flexible (PNG ou JPG)
        if image_path.lower().endswith('.png'):
            img = tf.image.decode_png(img, channels=3)
        else:
            img = tf.image.decode_jpeg(img, channels=3)
        
        # Redimensionner
        img = tf.image.resize(img, target_size)
        
        # Normaliser
        img = tf.cast(img, tf.float32) / 255.0
        
        # Ajouter dimension batch
        img = tf.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {image_path}: {e}")
        return None

def postprocess_mask(pred_mask, threshold=0.5):
    """Post-traite le masque prédit"""
    # Binarisation avec seuil
    binary_mask = tf.where(pred_mask > threshold, 1.0, 0.0)
    
    # Suppression du bruit (optionnel)
    # binary_mask = tf.nn.erosion2d(binary_mask, filters=tf.ones((3,3,1,1)), strides=[1,1,1,1], padding="SAME", data_format="NHWC", dilations=[1,1,1,1])
    # binary_mask = tf.nn.dilation2d(binary_mask, filters=tf.ones((3,3,1,1)), strides=[1,1,1,1], padding="SAME", data_format="NHWC", dilations=[1,1,1,1])
    
    return binary_mask

def predict_single_image(model, image_path, threshold=0.5):
    """Prédit le masque pour une seule image"""
    print(f"🖼️  Traitement: {os.path.basename(image_path)}")
    
    # Charger et préprocesser
    input_image = load_and_preprocess_image(image_path)
    if input_image is None:
        return None, None
    
    # Prédiction
    pred_mask = model.predict(input_image, verbose=0)[0]
    
    # Post-traitement
    processed_mask = postprocess_mask(pred_mask, threshold)
    
    return input_image, processed_mask

def batch_predict(model, image_dir, threshold=0.5, max_images=None):
    """Traite un lot d'images"""
    # Chercher les images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_paths:
        print(f"❌ Aucune image trouvée dans {image_dir}")
        return []
    
    # Limiter le nombre d'images si spécifié
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"📊 {len(image_paths)} image(s) à traiter")
    
    results = []
    for i, image_path in enumerate(image_paths):
        print(f"⏳ [{i+1}/{len(image_paths)}] ", end="")
        
        input_img, pred_mask = predict_single_image(model, image_path, threshold)
        if input_img is not None:
            results.append({
                'path': image_path,
                'input': input_img,
                'mask': pred_mask
            })
    
    return results

def visualize_results(results, save_dir="results", show_plots=True):
    """Visualise et sauvegarde les résultats"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i, result in enumerate(results[:6]):  # Limite à 6 pour l'affichage
        plt.figure(figsize=(12, 4))
        
        # Image originale
        plt.subplot(1, 3, 1)
        plt.title("🖼️ Image originale", fontweight='bold')
        plt.imshow(tf.squeeze(result['input']))
        plt.axis('off')
        
        # Masque prédit
        plt.subplot(1, 3, 2)
        plt.title("🎯 Masque prédit", fontweight='bold')
        plt.imshow(tf.squeeze(result['mask']), cmap='gray')
        plt.axis('off')
        
        # Superposition
        plt.subplot(1, 3, 3)
        plt.title("🔍 Superposition", fontweight='bold')
        
        # Image de base en transparence
        plt.imshow(tf.squeeze(result['input']), alpha=0.7)
        
        # Masque en rouge semi-transparent
        mask_colored = tf.squeeze(result['mask'])
        plt.imshow(mask_colored, cmap='Reds', alpha=0.5)
        plt.axis('off')
        
        plt.tight_layout()
        
        # Sauvegarder
        filename = f"result_{i+1}_{os.path.basename(result['path']).split('.')[0]}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    print(f"💾 Résultats sauvegardés dans: {save_dir}")

def calculate_metrics(results):
    """Calcule des métriques sur les résultats"""
    if not results:
        return
    
    print("\n📊 MÉTRIQUES DES PRÉDICTIONS")
    print("="*40)
    
    total_pixels = 0
    predicted_pixels = 0
    
    for result in results:
        mask = tf.squeeze(result['mask'])
        total = tf.size(mask).numpy()
        predicted = tf.reduce_sum(mask).numpy()
        
        total_pixels += total
        predicted_pixels += predicted
        
        coverage = (predicted / total) * 100
        print(f"📁 {os.path.basename(result['path'])[:20]:20} - Couverture: {coverage:.1f}%")
    
    overall_coverage = (predicted_pixels / total_pixels) * 100
    print(f"\n📈 Couverture moyenne: {overall_coverage:.1f}%")
    print(f"🔢 Total pixels traités: {total_pixels:,}")
    print(f"🎯 Pixels détectés: {predicted_pixels:,}")

# ===== EXÉCUTION PRINCIPALE =====
def main():
    print("\n🎛️ SÉLECTION DU MODE D'EXÉCUTION")
    print("="*40)
    print("1. 🖼️  Image unique")
    print("2. 📁 Dossier d'images") 
    print("3. 🧪 Image de test par défaut")
    
    try:
        choice = input("\nChoix (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
        return
    
    # Paramètres de prédiction
    threshold = 0.5
    
    try:
        threshold = float(input(f"Seuil de binarisation (défaut {threshold}): ") or threshold)
    except:
        print(f"⚠️ Seuil invalide, utilisation de {threshold}")
    
    if choice == "1":
        # Image unique
        image_path = input("Chemin de l'image: ").strip()
        if not os.path.exists(image_path):
            print(f"❌ Fichier non trouvé: {image_path}")
            return
        
        input_img, pred_mask = predict_single_image(model, image_path, threshold)
        if input_img is not None:
            results = [{'path': image_path, 'input': input_img, 'mask': pred_mask}]
            visualize_results(results)
            calculate_metrics(results)
    
    elif choice == "2":
        # Dossier d'images
        image_dir = input("Chemin du dossier: ").strip() or "dataset/CapturedImages"
        if not os.path.exists(image_dir):
            print(f"❌ Dossier non trouvé: {image_dir}")
            return
        
        max_images = input("Nombre max d'images (vide = toutes): ").strip()
        max_images = int(max_images) if max_images else None
        
        results = batch_predict(model, image_dir, threshold, max_images)
        if results:
            visualize_results(results, show_plots=False)  # Pas d'affichage pour lots
            calculate_metrics(results)
    
    elif choice == "3":
        # Test par défaut
        default_paths = [
            "dataset/CapturedImages/image_0_image_1000.png",
            "dataset/CapturedImages/image_0.png", 
            "test_image.png"
        ]
        
        test_path = None
        for path in default_paths:
            if os.path.exists(path):
                test_path = path
                break
        
        if not test_path:
            print("❌ Aucune image de test trouvée")
            print("Images recherchées:")
            for path in default_paths:
                print(f"   - {path}")
            return
        
        input_img, pred_mask = predict_single_image(model, test_path, threshold)
        if input_img is not None:
            results = [{'path': test_path, 'input': input_img, 'mask': pred_mask}]
            visualize_results(results)
            calculate_metrics(results)
    
    else:
        print("❌ Choix invalide")

# ===== FONCTIONS AVANCÉES =====
def benchmark_model(model, num_predictions=10):
    """Teste la vitesse du modèle"""
    print(f"\n⚡ BENCHMARK - {num_predictions} prédictions")
    print("="*40)
    
    # Créer une image factice
    dummy_image = tf.random.normal((1, 128, 128, 3))
    
    # Premier appel (compilation)
    _ = model.predict(dummy_image, verbose=0)
    
    # Mesurer la vitesse
    start_time = datetime.now()
    
    for i in range(num_predictions):
        _ = model.predict(dummy_image, verbose=0)
        if (i + 1) % 5 == 0:
            print(f"⏳ {i+1}/{num_predictions} prédictions...")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"⚡ Temps total: {duration:.2f}s")
    print(f"📊 Temps par prédiction: {duration/num_predictions:.3f}s")
    print(f"🔥 FPS: {num_predictions/duration:.1f}")

def interactive_threshold_tuning():
    """Interface interactive pour ajuster le seuil"""
    print("\n🎛️ AJUSTEMENT INTERACTIF DU SEUIL")
    print("="*40)
    
    # Sélectionner une image de test
    test_dir = "dataset/CapturedImages"
    if not os.path.exists(test_dir):
        print(f"❌ Dossier de test non trouvé: {test_dir}")
        return
    
    images = glob.glob(os.path.join(test_dir, "*.png"))
    if not images:
        print(f"❌ Aucune image de test dans: {test_dir}")
        return
    
    test_image = images[0]
    print(f"🖼️ Image de test: {os.path.basename(test_image)}")
    
    # Charger l'image
    input_img = load_and_preprocess_image(test_image)
    if input_img is None:
        return
    
    # Prédiction de base
    pred_mask_raw = model.predict(input_img, verbose=0)[0]
    
    while True:
        try:
            threshold = float(input(f"\n🎯 Seuil (0.0-1.0, 'q' pour quitter): "))
            
            if threshold < 0 or threshold > 1:
                print("⚠️ Seuil doit être entre 0.0 et 1.0")
                continue
            
            # Appliquer le seuil
            processed_mask = postprocess_mask(pred_mask_raw, threshold)
            
            # Calculer la couverture
            coverage = (tf.reduce_sum(processed_mask) / tf.size(processed_mask)).numpy() * 100
            
            print(f"📊 Couverture avec seuil {threshold}: {coverage:.1f}%")
            
            # Visualisation rapide
            plt.figure(figsize=(10, 3))
            
            plt.subplot(1, 3, 1)
            plt.title("Image")
            plt.imshow(tf.squeeze(input_img))
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title(f"Masque brut")
            plt.imshow(tf.squeeze(pred_mask_raw), cmap='gray')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title(f"Seuil {threshold}")
            plt.imshow(tf.squeeze(processed_mask), cmap='gray')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except (ValueError, KeyboardInterrupt):
            break
    
    print("👋 Fin de l'ajustement")

if __name__ == "__main__":
    print("\n🔧 FONCTIONNALITÉS DISPONIBLES")
    print("="*40)
    print("1. 🎯 Prédiction standard")
    print("2. ⚡ Benchmark de vitesse")
    print("3. 🎛️ Ajustement interactif du seuil")
    
    try:
        mode = input("\nMode (1-3, Entrée=1): ").strip() or "1"
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
        exit()
    
    if mode == "1":
        main()
    elif mode == "2":
        benchmark_model(model)
    elif mode == "3":
        interactive_threshold_tuning()
    else:
        print("Mode par défaut")
        main()
    
    print("\n✅ Terminé!")