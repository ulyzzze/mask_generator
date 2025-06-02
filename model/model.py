import tensorflow as tf
from tensorflow.keras import layers, models

def unet_model(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # ... continue jusqu'à une profondeur raisonnable

    u = layers.UpSampling2D((2, 2))(c2)  # ou plus haut selon profondeur
    u = layers.Concatenate()([u, c1])
    c3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u)
    c3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c3)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)
    return models.Model(inputs, outputs)

def get_compiled_unet():
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model