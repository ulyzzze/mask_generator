from model import get_compiled_unet
from load_data import load_data
import numpy as np

model = get_compiled_unet()

# Normalise les données :
X, Y = load_data("dataset/CapturedImages/", "dataset/mask/")
X = X / 255.0
Y = np.expand_dims(Y, axis=-1)  # pour avoir la forme (batch, h, w, 1)

model.fit(X, Y, epochs=10, batch_size=16, validation_split=0.1)

model.save("model.h5")