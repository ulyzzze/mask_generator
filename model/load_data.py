import tensorflow as tf
import os

IMG_HEIGHT = 128
IMG_WIDTH = 128

def process_path(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method="nearest")

    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32)  # Changé en float32 pour plus de cohérence
    mask = tf.where(mask > 0.5, 1.0, 0.0)  # Seuil ajusté et valeurs en float

    return image, mask

def load_dataset(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Optimisation: mapping avec plus de parallélisme et non-déterministe pour la vitesse
    dataset = dataset.map(
        process_path, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False  # Plus rapide mais ordre non garanti
    )
    
    return dataset

def augment(image, mask):
    # Augmentations rapides et efficaces
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    # Rotation par 90° (plus rapide que les rotations arbitraires)
    if tf.random.uniform(()) > 0.7:
        k = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        mask = tf.image.rot90(mask, k)
    
    # Ajustement de luminosité léger
    if tf.random.uniform(()) > 0.8:
        image = tf.image.adjust_brightness(image, delta=tf.random.uniform([], -0.1, 0.1))
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def get_dataset(batch_size=8):  # Batch size réduit pour votre configuration RAM
    image_dir = "../dataset/CapturedImages"
    mask_dir = "../dataset/mask"
    
    # Charger le dataset de base
    train_ds = load_dataset(image_dir, mask_dir)
    
    # CRITIQUE: Cache en mémoire si le dataset est petit
    # Attention: consomme de la RAM, retirez cette ligne si problème de mémoire
    train_ds = train_ds.cache()
    
    # Appliquer les augmentations avec parallélisme
    train_ds = train_ds.map(
        augment, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    
    # Pipeline optimisé pour la performance
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.shuffle(buffer_size=100)  # Buffer réduit pour économiser la RAM
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)  # Pré-chargement en arrière-plan
    
    return train_ds