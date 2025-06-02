from model import get_compiled_unet
from load_data import get_dataset
import tensorflow as tf

# Configuration GPU (si disponible)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponible: {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

# Charger les datasets
train_ds, val_ds = get_dataset(validation_split=0.2)

# Charger et compiler le modèle
model = get_compiled_unet()
model.summary()

# Callbacks pour l'entraînement
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_line_segment_model.h5",
        monitor='val_dice_coefficient',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
]

# Entraîner le modèle avec plus d'époques
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,  # Plus d'époques pour un meilleur apprentissage
    callbacks=callbacks,
    verbose=1
)

# Sauvegarder le modèle final
model.save("line_segment_model_final.h5")

# Afficher les métriques finales
print("\n=== Métriques finales ===")
print(f"Accuracy finale: {history.history['accuracy'][-1]:.4f}")
print(f"Val Accuracy finale: {history.history['val_accuracy'][-1]:.4f}")
print(f"Dice coefficient final: {history.history['dice_coefficient'][-1]:.4f}")
print(f"Val Dice coefficient final: {history.history['val_dice_coefficient'][-1]:.4f}")

# Tracer les courbes d'apprentissage
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

# Perte
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Dice Coefficient
plt.subplot(1, 3, 3)
plt.plot(history.history['dice_coefficient'], label='Train Dice')
plt.plot(history.history['val_dice_coefficient'], label='Val Dice')
plt.title('Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()