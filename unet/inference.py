import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from unet.carvana_dataset import CarvanaDataset
from unet.unet import UNet

def pred_show_image_grid(data_path, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = CarvanaDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0]=0
        pred_mask[pred_mask > 0]=1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3*len(image_dataset)+1):
       fig.add_subplot(3, len(image_dataset), i)
       plt.imshow(images[i-1], cmap="gray")
    plt.show()

def tensor_to_numpy_mask(tensor_mask):
    """
    Convertit un tensor mask en numpy array pour le raycast
    
    Args:
        tensor_mask: Tensor PyTorch du masque
    
    Returns:
        numpy_mask: Masque en format numpy uint8
    """
    if tensor_mask.dim() == 3:
        tensor_mask = tensor_mask.squeeze()
    
    numpy_mask = tensor_mask.numpy()
    numpy_mask = (numpy_mask * 255).astype(np.uint8)
    
    return numpy_mask

def single_image_inference(image_path, model_path, device="cpu", show_plot=True):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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
