import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from carvana_dataset import CarvanaDataset
from unet import UNet

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


def single_image_inference(image_path, model_path, device="cpu"):
    # Charger le modèle
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # important pour désactiver dropout, batchnorm en mode entraînement

    # Transformation de l'image (redimensionner + convertir en tensor)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Charger et transformer l'image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # ajoute dimension batch

    # Faire la prédiction
    with torch.no_grad():  # pas de calcul des gradients en inférence
        pred_mask = model(img_tensor)

    # Post-traitement masque
    pred_mask = pred_mask.squeeze(0).cpu()  # enlever dimension batch et ramener CPU
    pred_mask = pred_mask.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # Binarisation simple (seuil à 0)
    pred_mask = (pred_mask > 0).float()

    # Affichage image + masque prédit
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(pred_mask.squeeze(), cmap="gray")  # masque en niveaux de gris
    axes[1].set_title("Masque prédit")
    axes[1].axis("off")

    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    image_path = "../dataset/CapturedImages/image_0_image_77.png"
    model_path = "./unet.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    single_image_inference(image_path, model_path, device)