import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Importer nos modules optimisés
from model import get_compiled_unet, print_model_info
from load_data import get_dataset

print("🚀 ENTRAÎNEMENT OPTIMISÉ POUR VOTRE CONFIGURATION")
print("="*60)

# ===== CONFIGURATION SYSTÈME =====
print("⚙️  Configuration du système...")

# Optimiser l'utilisation du CPU (important pour Intel Iris Xe)
tf.config.threading.set_intra_op_parallelism_threads(0)  # Utilise tous les cœurs
tf.config.threading.set_inter_op_parallelism_threads(0)  # Parallélisme entre opérations

# Vérifier les devices disponibles
print("🖥️  Devices disponibles:")
for device in tf.config.list_physical_devices():
    print(f"   - {device}")

# Optimiser la mémoire GPU si disponible
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Croissance mémoire GPU activée")
    except RuntimeError as e:
        print(f"⚠️  Erreur config GPU: {e}")
else:
    print("ℹ️  Pas de GPU dédié détecté - utilisation CPU + Intel Iris Xe")

# ===== PARAMÈTRES D'ENTRAÎNEMENT =====
BATCH_SIZE = 8          # Optimisé pour 16GB RAM
EPOCHS = 50             # Réduit de 100 à 50 pour tester plus vite
VALIDATION_SPLIT = 0.2  # 20% pour validation

print(f"📊 Paramètres:")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - Épochs: {EPOCHS}")
print(f"   - Validation: {VALIDATION_SPLIT*100:.0f}%")

# ===== CHARGEMENT DES DONNÉES =====
print("\n📁 Chargement du dataset...")
try:
    train_ds = get_dataset(batch_size=BATCH_SIZE)
    print("✅ Dataset chargé avec succès")
    
    # Calculer le nombre d'échantillons (approximatif)
    sample_count = 0
    for batch in train_ds.take(10):  # Échantillon pour estimation
        sample_count += batch[0].shape[0]
    estimated_total = sample_count * 10  # Estimation grossière
    print(f"📈 Échantillons estimés: ~{estimated_total}")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    print("Vérifiez que les dossiers 'dataset/CapturedImages' et 'dataset/mask' existent")
    exit(1)

# ===== CRÉATION DU MODÈLE =====
print("\n🏗️  Création du modèle...")
model = get_compiled_unet(use_light_version=True)  # Version allégée recommandée

# Afficher les informations du modèle
print_model_info(model)

# ===== CALLBACKS INTELLIGENTS =====
print("🎛️  Configuration des callbacks...")

# Créer le dossier pour les sauvegardes
os.makedirs("checkpoints", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

callbacks = [
    # Réduction automatique du learning rate
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,              # Divise par 2
        patience=5,              # Attend 5 époques
        min_lr=1e-7,            # Learning rate minimum
        verbose=1,
        cooldown=2               # Attend 2 époques avant nouvelle réduction
    ),
    
    # Arrêt anticipé si pas d'amélioration
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,             # Patience plus élevée
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001         # Amélioration minimale requise
    ),
    
    # Sauvegarde du meilleur modèle
    tf.keras.callbacks.ModelCheckpoint(
        f'checkpoints/best_model_{timestamp}.h5',
        monitor='loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='min'
    ),
    
    # Logging détaillé
    tf.keras.callbacks.CSVLogger(
        f'checkpoints/training_log_{timestamp}.csv',
        separator=',',
        append=False
    ),
    
    # Réduction progressive du learning rate (backup)
    tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.001 * (0.95 ** epoch),
        verbose=0
    )
]

print(f"✅ {len(callbacks)} callbacks configurés")

# ===== ENTRAÎNEMENT =====
print("\n" + "="*60)
print("🔥 DÉBUT DE L'ENTRAÎNEMENT")
print("="*60)

try:
    # Mesurer le temps d'entraînement
    start_time = datetime.now()
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,                    # Affichage détaillé
        workers=4,                    # Parallélisme pour le chargement des données
        use_multiprocessing=False,    # Évite les problèmes sur certains systèmes
        max_queue_size=10            # Limite la queue pour économiser la RAM
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    print("\n" + "="*60)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print(f"⏱️  Durée: {training_duration}")
    print(f"📈 Époques réalisées: {len(history.history['loss'])}")
    print("="*60)
    
except KeyboardInterrupt:
    print("\n⚠️  Entraînement interrompu par l'utilisateur")
    print("💾 Sauvegarde du modèle actuel...")
    model.save(f"model_interrupted_{timestamp}.h5")
    
except Exception as e:
    print(f"\n❌ Erreur pendant l'entraînement: {e}")
    print("💾 Tentative de sauvegarde...")
    try:
        model.save(f"model_error_{timestamp}.h5")
        print("✅ Modèle sauvegardé malgré l'erreur")
    except:
        print("❌ Impossible de sauvegarder")

# ===== SAUVEGARDE FINALE =====
print("\n💾 Sauvegarde du modèle final...")
try:
    final_model_path = f"line_segment_model_optimized_{timestamp}.h5"
    model.save(final_model_path)
    print(f"✅ Modèle sauvegardé: {final_model_path}")
    
    # Sauvegarde de l'historique
    np.save(f'checkpoints/history_{timestamp}.npy', history.history)
    print(f"📊 Historique sauvegardé: checkpoints/history_{timestamp}.npy")
    
except Exception as e:
    print(f"❌ Erreur lors de la sauvegarde: {e}")

# ===== VISUALISATION DES RÉSULTATS =====
print("\n📊 Génération des graphiques de performance...")

try:
    plt.style.use('default')  # Style propre
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], 'b-', label='Loss', linewidth=2)
    axes[0, 0].set_title('📉 Évolution de la Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Époque')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], 'g-', label='Accuracy', linewidth=2)
    axes[0, 1].set_title('📈 Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Époque')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], 'r-', label='Precision', linewidth=2)
        axes[1, 0].set_title('🎯 Évolution de la Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Époque')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], 'm-', label='Recall', linewidth=2)
        axes[1, 1].set_title('🔍 Évolution du Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Époque')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    graph_path = f'checkpoints/training_curves_{timestamp}.png'
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"📊 Graphiques sauvegardés: {graph_path}")
    
    plt.show()
    
except Exception as e:
    print(f"⚠️  Erreur lors de la génération des graphiques: {e}")

# ===== RÉSUMÉ FINAL =====
print("\n" + "="*60)
print("🎉 ENTRAÎNEMENT COMPLÉTÉ")
print("="*60)

if 'history' in locals():
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    
    print(f"📊 Résultats finaux:")
    print(f"   - Loss finale: {final_loss:.4f}")
    print(f"   - Accuracy finale: {final_accuracy:.4f}")
    
    if 'precision' in history.history:
        final_precision = history.history['precision'][-1]
        print(f"   - Precision finale: {final_precision:.4f}")
    
    if 'recall' in history.history:
        final_recall = history.history['recall'][-1]
        print(f"   - Recall finale: {final_recall:.4f}")

print(f"\n🎯 Prochaines étapes:")
print(f"   1. Testez le modèle avec apply_model.py")
print(f"   2. Vérifiez les résultats visuels")
print(f"   3. Ajustez les paramètres si nécessaire")

print("\n💡 Conseils d'optimisation:")
print("   - Si trop lent: réduisez IMG_HEIGHT/WIDTH à 96")
print("   - Si manque de RAM: réduisez BATCH_SIZE à 4")
print("   - Si sous-apprentissage: augmentez EPOCHS")

print("="*60)