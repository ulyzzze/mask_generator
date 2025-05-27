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
    mask = tf.cast(mask, tf.uint8)
    mask = tf.where(mask > 0, 1, 0)  # binaire

    return image, mask

def load_dataset(image_dir, mask_dir):
    image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
    mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask

def get_dataset():
    train_ds = load_dataset("dataset/CapturedImages", "dataset/mask")
    train_ds = train_ds.map(augment)
    train_ds = train_ds.batch(32).shuffle(100).prefetch(tf.data.AUTOTUNE)
    return train_ds
