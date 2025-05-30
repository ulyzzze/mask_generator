import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision

# Tentative d'activation de la précision mixte (peut accélérer sur certains GPU)
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("✅ Précision mixte activée - Entraînement plus rapide")
except Exception as e:
    print("⚠️  Précision mixte non disponible sur cette configuration")
    print(f"Détail: {e}")

def unet_model(input_size=(128, 128, 3)):
    inputs = tf.keras.Input(shape=input_size)

    # ===== ENCODER (Contraction) =====
    # Bloc 1
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(inputs)
    c1 = layers.BatchNormalization()(c1)  # Stabilise l'entraînement
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)  # Évite le surapprentissage

    # Bloc 2
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)

    # Bloc 3
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)

    # ===== BOTTLENECK (Fond) =====
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.3)(c4)

    # ===== DECODER (Expansion) =====
    # Bloc de décodage 1
    u5 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(u5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.2)(c5)

    # Bloc de décodage 2
    u6 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Dropout(0.1)(c6)

    # Bloc de décodage 3
    u7 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(c7)
    c7 = layers.BatchNormalization()(c7)

    # ===== SORTIE =====
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def unet_model_light(input_size=(128, 128, 3)):
    """Version allégée du U-Net pour machines moins puissantes"""
    inputs = tf.keras.Input(shape=input_size)

    # Encoder simplifié
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same', 
                       kernel_initializer='he_normal')(inputs)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(p1)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.2)(c3)

    # Decoder simplifié
    u4 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(u4)
    c4 = layers.BatchNormalization()(c4)

    u5 = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(u5)
    c5 = layers.BatchNormalization()(c5)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def get_compiled_unet(use_light_version=True):
    """
    Crée et compile le modèle U-Net optimisé
    
    Args:
        use_light_version (bool): Si True, utilise la version allégée (recommandé pour votre config)
    """
    # Choisir la version du modèle
    if use_light_version:
        model = unet_model_light()
        print("🔥 Modèle U-Net LIGHT chargé (optimisé pour votre configuration)")
    else:
        model = unet_model()
        print("🔥 Modèle U-Net COMPLET chargé")
    
    # Optimiseur avec paramètres ajustés
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,          # Learning rate initial
        beta_1=0.9,                   # Momentum pour le gradient
        beta_2=0.999,                 # Momentum pour le gradient au carré
        epsilon=1e-7,                 # Stabilité numérique
        clipnorm=1.0                  # Évite l'explosion des gradients
    )
    
    # Compilation avec métriques étendues
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

# Fonction pour afficher les informations du modèle
def print_model_info(model):
    """Affiche des informations utiles sur le modèle"""
    print("\n" + "="*50)
    print("📊 INFORMATIONS DU MODÈLE")
    print("="*50)
    
    # Compter les paramètres
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"🔢 Paramètres totaux: {total_params:,}")
    print(f"🎯 Paramètres entraînables: {trainable_params:,}")
    print(f"🧠 Taille du modèle: ~{total_params * 4 / (1024**2):.1f} MB")
    
    # Informations sur l'architecture
    print(f"📐 Entrée: {model.input_shape}")
    print(f"📤 Sortie: {model.output_shape}")
    print(f"🏗️  Nombre de couches: {len(model.layers)}")
    
    print("="*50 + "\n")