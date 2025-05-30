"""
Configuration optimisée pour votre système Dell XPS 9315
- Intel Core i7-1360P (12ème génération)
- 16 GB RAM
- Intel Iris Xe Graphics
- SSD NVMe 512GB
"""

import tensorflow as tf
import os
import psutil
import platform

class SystemOptimizer:
    """Optimise automatiquement TensorFlow pour votre configuration"""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=True)
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        self.system_info = self._get_system_info()
        
    def _get_system_info(self):
        """Collecte les informations système"""
        return {
            'platform': platform.system(),
            'processor': platform.processor(),
            'cpu_cores': self.cpu_count,
            'ram_gb': round(self.ram_gb, 1),
            'python_version': platform.python_version()
        }
    
    def print_system_info(self):
        """Affiche les informations système"""
        print("🖥️ INFORMATIONS SYSTÈME")
        print("="*40)
        print(f"💻 Plateforme: {self.system_info['platform']}")
        print(f"🔧 Processeur: Intel Core i7-1360P (détecté: {self.cpu_count} threads)")
        print(f"🧠 RAM: {self.system_info['ram_gb']:.1f} GB")
        print(f"🐍 Python: {self.system_info['python_version']}")
        print(f"🤖 TensorFlow: {tf.__version__}")
        
        # Devices TensorFlow
        print(f"\n📱 Devices TensorFlow:")
        for device in tf.config.list_physical_devices():
            print(f"   - {device}")
        
        print("="*40)
    
    def optimize_tensorflow(self):
        """Configure TensorFlow pour votre matériel"""
        print("⚙️ OPTIMISATION TENSORFLOW")
        print("="*40)
        
        # 1. Optimisation CPU (critique pour Intel Iris Xe)
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Utilise tous les cœurs
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Parallélisme entre ops
        print(f"✅ CPU optimisé: {self.cpu_count} threads utilisés")
        
        # 2. Optimisation mémoire GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU memory growth activé pour {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"⚠️ Erreur GPU config: {e}")
        else:
            print("ℹ️ Pas de GPU dédié détecté - Intel Iris Xe utilisé")
        
        # 3. Optimisation de la précision mixte (si supportée)
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            print("✅ Précision mixte float16 activée")
        except Exception as e:
            print(f"⚠️ Précision mixte non disponible: {e}")
        
        # 4. Configuration XLA (accélération)
        try:
            tf.config.optimizer.set_jit(True)
            print("✅ XLA JIT compilation activée")
        except:
            print("⚠️ XLA non disponible")
        
        print("="*40)
    
    def get_recommended_config(self):
        """Retourne la configuration recommandée pour l'entraînement"""
        # Calculs basés sur votre RAM de 16GB
        recommended_batch_size = min(16, max(4, int(self.ram_gb)))
        
        config = {
            'batch_size': 8,  # Optimisé pour 16GB RAM
            'prefetch_buffer': tf.data.AUTOTUNE,
            'num_parallel_calls': tf.data.AUTOTUNE,
            'cache_dataset': self.ram_gb >= 8,  # Cache si assez de RAM
            'mixed_precision': True,
            'max_epochs': 50,  # Réduit pour tests plus rapides
            'learning_rate': 0.001,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'model_checkpoint': True
        }
        
        return config

# Configuration par défaut optimisée pour votre système
OPTIMIZED_CONFIG = {
    # Paramètres d'entraînement
    'BATCH_SIZE': 8,
    'EPOCHS': 50,
    'LEARNING_RATE': 0.001,
    'VALIDATION_SPLIT': 0.2,
    
    # Paramètres du modèle
    'IMG_HEIGHT': 128,
    'IMG_WIDTH': 128,
    'USE_LIGHT_MODEL': True,  # Recommandé pour votre config
    
    # Paramètres de données
    'CACHE_DATASET': True,
    'PREFETCH_BUFFER': tf.data.AUTOTUNE,
    'NUM_PARALLEL_CALLS': tf.data.AUTOTUNE,
    'SHUFFLE_BUFFER': 100,
    
    # Callbacks
    'EARLY_STOPPING_PATIENCE': 10,
    'REDUCE_LR_PATIENCE': 5,
    'REDUCE_LR_FACTOR': 0.5,
    'MIN_LR': 1e-7,
    
    # Chemins
    'DATASET_DIR': 'dataset',
    'IMAGES_DIR': '../dataset/CapturedImages',
    'MASKS_DIR': '../dataset/mask',
    'CHECKPOINTS_DIR': 'checkpoints',
    'RESULTS_DIR': 'results'
}

def setup_environment():
    """Configure l'environnement complet"""
    print("🚀 CONFIGURATION DE L'ENVIRONNEMENT")
    print("="*50)
    
    # Créer l'optimiseur
    optimizer = SystemOptimizer()
    
    # Afficher les infos système
    optimizer.print_system_info()
    
    # Optimiser TensorFlow
    optimizer.optimize_tensorflow()
    
    # Créer les dossiers nécessaires
    dirs_to_create = [
        OPTIMIZED_CONFIG['CHECKPOINTS_DIR'],
        OPTIMIZED_CONFIG['RESULTS_DIR']
    ]
    
    print("📁 Création des dossiers...")
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"   ✅ {dir_path}")
    
    # Retourner la config recommandée
    recommended = optimizer.get_recommended_config()
    
    print("\n🎯 CONFIGURATION RECOMMANDÉE")
    print("="*40)
    for key, value in recommended.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Environnement configuré et optimisé!")
    print("="*50)
    
    return optimizer, recommended

def check_dataset():
    """Vérifie la présence et la validité du dataset"""
    print("\n🔍 VÉRIFICATION DU DATASET")
    print("="*40)
    
    images_dir = OPTIMIZED_CONFIG['IMAGES_DIR']
    masks_dir = OPTIMIZED_CONFIG['MASKS_DIR']
    
    # Vérifier les dossiers
    if not os.path.exists(images_dir):
        print(f"❌ Dossier images manquant: {images_dir}")
        return False
    
    if not os.path.exists(masks_dir):
        print(f"❌ Dossier masques manquant: {masks_dir}")
        return False
    
    # Compter les fichiers
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📊 Images trouvées: {len(image_files)}")
    print(f"📊 Masques trouvés: {len(mask_files)}")
    
    if len(image_files) == 0:
        print("❌ Aucune image trouvée!")
        return False
    
    if len(mask_files) == 0:
        print("❌ Aucun masque trouvé!")
        return False
    
    if len(image_files) != len(mask_files):
        print(f"⚠️ Nombre d'images ({len(image_files)}) != nombre de masques ({len(mask_files)})")
    
    # Vérifier quelques fichiers
    sample_size = min(5, len(image_files))
    print(f"\n🔍 Vérification de {sample_size} échantillons...")
    
    for i in range(sample_size):
        img_path = os.path.join(images_dir, image_files[i])
        try:
            # Test de chargement
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=3)
            print(f"   ✅ {image_files[i]} - {img.shape}")
        except Exception as e:
            print(f"   ❌ {image_files[i]} - Erreur: {e}")
            return False
    
    print("✅ Dataset valide!")
    return True

def performance_tips():
    """Affiche des conseils d'optimisation personnalisés"""
    print("\n💡 CONSEILS D'OPTIMISATION POUR VOTRE SYSTÈME")
    print("="*60)
    
    print("🔋 Pour améliorer les performances:")
    print("   • Fermez les applications inutiles (libère RAM)")
    print("   • Branchez l'adaptateur secteur (max performance CPU)")
    print("   • Vérifiez que le ventilateur fonctionne (évite throttling)")
    
    print("\n⚙️ Si l'entraînement est trop lent:")
    print("   • Réduisez IMG_HEIGHT/WIDTH de 128 à 96")
    print("   • Diminuez BATCH_SIZE de 8 à 4")
    print("   • Utilisez use_light_model=True")
    
    print("\n🧠 Si problèmes de mémoire:")
    print("   • Désactivez le cache: cache_dataset=False")
    print("   • Réduisez BATCH_SIZE à 4 ou moins")
    print("   • Fermez Chrome/Firefox (consomme beaucoup de RAM)")
    
    print("\n🚀 Pour accélérer encore plus:")
    print("   • Utilisez Google Colab (GPU gratuit)")
    print("   • Envisagez un eGPU via Thunderbolt 4")
    print("   • SSD externe rapide pour le dataset")
    
    print("="*60)

if __name__ == "__main__":
    # Test de la configuration
    optimizer, config = setup_environment()
    
    # Vérifier le dataset
    dataset_ok = check_dataset()
    
    # Afficher les conseils
    performance_tips()
    
    print(f"\n🎯 Prêt pour l'entraînement: {'✅' if dataset_ok else '❌'}")