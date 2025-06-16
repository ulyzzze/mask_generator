import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def raycast(mask, num_rays=10, fov_degrees=90, max_length=200, show_plot=True):
    h, w = mask.shape
    origin = (w // 2, h - 10)

    fov_radians = math.radians(fov_degrees)
    start_angle = -fov_radians / 2
    angle_step = fov_radians / (num_rays - 1)

    distances = []
    hit_points = []

    for i in range(num_rays):
        angle = start_angle + i * angle_step + math.pi/2
        dx = math.cos(angle)
        dy = -math.sin(angle)

        hit = False
        for length in range(5, max_length):
            
            x = int(origin[0] + dx * length)
            y = int(origin[1] + dy * length)
            if x < 0 or x >= w or y < 0 or y >= h:
                break
            if mask[y, x] > 128:
                hit = True
                break
        distances.append(length)
        hit_points.append((x, y))

    if show_plot:
        plt.imshow(mask, cmap='gray')
        for pt in hit_points:
            plt.plot([mask.shape[1] // 2, pt[0]], [mask.shape[0] - 1, pt[1]], 'r-')
        plt.scatter(mask.shape[1] // 2, mask.shape[0] - 1, color='green')
        plt.show() 

    return distances, hit_points
