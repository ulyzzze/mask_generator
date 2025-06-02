import tensorflow as tf
import os

IMG_HEIGHT = 128
IMG_WIDTH = 128

def process_path(image_path, mask_path):
    # Lecture des images
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    # Redimensionnement
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method="nearest")

    # Normalisation de l'image
    image = tf.cast(image, tf.float32) / 255.0
    
    # Traitement du masque - plus robuste
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)  # Binarisation avec seuil
    
    return image, mask

def load_dataset(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
    
    print(f"Trouvé {len(image_paths)} images et {len(mask_paths)} masques")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def augment(image, mask):
    """Augmentation de données plus riche"""
    # Rotation aléatoire
    if tf.random.uniform(()) > 0.5:
        angle = tf.random.uniform((), -0.2, 0.2)  # ±11.5 degrés
        image = tf.image.rot90(image, k=tf.random.uniform((), 0, 4, dtype=tf.int32))
        mask = tf.image.rot90(mask, k=tf.random.uniform((), 0, 4, dtype=tf.int32))
    
    # Flip horizontal
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    
    # Flip vertical
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    
    # Ajustement de luminosité
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, 0.2)
    
    # Ajustement du contraste
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Saturation
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_saturation(image, 0.8, 1.2)
    
    # S'assurer que l'image reste dans [0,1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def get_dataset(validation_split=0.2):
    """Créer les datasets d'entraînement et de validation"""
    dataset = load_dataset("dataset/CapturedImages", "dataset/mask")
    
    # Compter le nombre total d'échantillons
    dataset_size = sum(1 for _ in dataset)
    print(f"Taille totale du dataset: {dataset_size}")
    
    # Mélanger le dataset
    dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    # Division train/validation
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)
    
    # Augmentation pour l'entraînement seulement
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Préparation des datasets
    train_ds = train_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(16).prefetch(tf.data.AUTOTUNE)
    
    print(f"Dataset d'entraînement: {train_size} échantillons")
    print(f"Dataset de validation: {val_size} échantillons")
    
    return train_ds, val_ds