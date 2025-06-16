import torch
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import time

from raycast.raycast import raycast
from unet.inference import single_image_inference

def lidars_from_predicted_mask(image_path, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = UNet(in_channels=3, num_classes=1).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    # model.eval()
    #start_time = time.time()
    predicted_mask = single_image_inference(image_path, model, device, False)      
    distances, points = raycast(predicted_mask, num_rays=10, fov_degrees=187, show_plot=False)    

    return distances, points 
    #print(f"{distances}")

    #end_time = time.time()
    #execution_time = end_time - start_time
    #print(f"Temps d'exécution: {execution_time:.4f} secondes")

#lidars_from_predicted_mask("dataset/CapturedImages/image_0_image_2000.png")