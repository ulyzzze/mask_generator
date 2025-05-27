from model import get_compiled_unet
from load_data import get_dataset

# Charger le dataset
train_ds = get_dataset()

# Charger et compiler le modèle
model = get_compiled_unet()

# Entraîner le modèle   
model.fit(train_ds, epochs=10)

# Sauvegarder le modèle
model.save("line_segment_model.h5")
