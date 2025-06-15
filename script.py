import torch
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

from raycast.raycast import raycast
from unet.inference import single_image_inference

if __name__ == "__main__":
    #image_path = "unet/imagesOnCar/23_original_img.jpg"
    image_path = "dataset/CapturedImages/image_0_image_1.png"
    model_path = "trainedIA/ia/2epochs.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predicted_mask = single_image_inference(image_path, model_path, device)

    #mask = cv2.imread("dataset/mask/image_0_mask_2865.png", cv2.IMREAD_GRAYSCALE)
    distances, points = raycast(predicted_mask, num_rays=10, fov_degrees=120)

    print(distances)

    plt.imshow(predicted_mask, cmap='gray')
    for pt in points:
        plt.plot([predicted_mask.shape[1] // 2, pt[0]], [predicted_mask.shape[0] - 1, pt[1]], 'r-')
    plt.scatter(predicted_mask.shape[1] // 2, predicted_mask.shape[0] - 1, color='green')
    plt.show()