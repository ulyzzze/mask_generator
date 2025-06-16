import numpy as np
import platform
import time
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import joblib

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet.unet import UNet
from script import lidars_from_predicted_mask

# Chargement du modèle Random Forest pré-entraîné
rf_model = joblib.load("rf_model.pkl")
print("Modèle Random Forest chargé depuis rf_model.pkl")

# Détection du système pour déterminer le binaire Unity
if platform.system() == "Linux":
    binary_path = "./BuildLinux/LinuxBinary/RacingSimulator.x86_64"
else:
    raise EnvironmentError("OS non pris en charge.")

# Lancement de l’environnement Unity (sans fichier si déjà lancé à part)
env = UnityEnvironment(file_name=None, base_port=5004, worker_id=0)
env.reset()

# Récupération des infos de comportement
behavior_name = list(env.behavior_specs.keys())[0]
spec = env.behavior_specs[behavior_name]

print("Simulation démarrée. Le véhicule est autonome")

image_folder = "/home/ulysse/Téléchargements/RacingSimulator/Assets/CapturedImages"
last_processed_image = 0
current_lidar = None

#Load Unet model
model_path = "../trainedIA/ia/2epochs.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=3, num_classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

try:
    image_counter = 1
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        for agent_id in decision_steps:
            # Vérifier s'il y a une nouvelle image disponible
            expected_image_path = os.path.join(image_folder, f"image_0_image_{image_counter}.png")

            if os.path.exists(expected_image_path) and image_counter > last_processed_image:
                print(f"Nouvelle image détectée: {expected_image_path}")
                try:
                    # Appliquer la fonction de génération de lidars
                    distances, points = lidars_from_predicted_mask(expected_image_path, model)
                    last_processed_image = image_counter
                    
                    # Utiliser les lidars générés par le modèle de vision
                    obs = np.array(distances[:10])  # Prendre les 10 premiers
                    print(f"Utilisation des lidars générés: {obs}")
                    
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    current_lidar = obs_tensor

                    # Prédiction via modèle
                    pred = rf_model.predict(obs_tensor.numpy())
                    steering = np.clip(pred[0, 0], -1.0, 1.0)
                    throttle = np.clip(pred[0, 1], -1.0, 1.0)

                    # Application de l'action à l'environnement
                    action = np.array([[throttle, steering]], dtype=np.float32)
                    env.set_actions(behavior_name, ActionTuple(continuous=action))
                    
                    image_counter += 1
                    
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image: {e}")
            else:
                # Pas de nouvelle image disponible, attendre sans action
                action = np.array([[throttle, steering]], dtype=np.float32)
                env.set_actions(behavior_name, ActionTuple(continuous=action))
                print(f"En attente de l'image: image_0_image_{image_counter}.png")

        env.step()
        time.sleep(0.00)
        

except KeyboardInterrupt:
    print("Arrêt manuel de la simulation.")

finally:
    env.close()
    print("Simulation terminée.")
