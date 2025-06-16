import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

def tensor_to_numpy_mask(tensor_mask):
    if tensor_mask.dim() == 3:
        tensor_mask = tensor_mask.squeeze()
    
    numpy_mask = tensor_mask.numpy()
    numpy_mask = (numpy_mask * 255).astype(np.uint8)
    
    return numpy_mask

def single_image_inference(image_path, model, device="cpu", show_plot=True):
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(img_tensor)

    pred_mask = pred_mask.squeeze(0).cpu()
    pred_mask = pred_mask.permute(1, 2, 0)

    pred_mask = (pred_mask > 0).float()

    mask_numpy = tensor_to_numpy_mask(pred_mask)

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        axes[1].imshow(pred_mask.squeeze(), cmap="gray")
        axes[1].set_title("Masque prédit")
        axes[1].axis("off")

        plt.show()
    
    return mask_numpy
