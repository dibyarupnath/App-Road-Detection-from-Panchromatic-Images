import os
import time
from flask import Flask, request, render_template, url_for
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import CustomRoadData
from roadseg_nn import RoadSegNN
from segnet import SegNet
from utils import seeding, epoch_time
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import sys
import matplotlib.pyplot as plt


def display_image(image_path):
    try:
        # Open an image file
        image = Image.open(image_path)
        # Display image
        image.show()
    except IOError:
        print("Unable to open image file.")


def detect_rd(model_type, img_path):
    # # Get the selected model type
    # model_type = sys.argv[1]

    # # Get the sat image
    # img_path = sys.argv[2]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    H = 256
    W = 256
    size = (H, W)

    # For input to the Model (The image is being normalised)
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(0.2826, 0.2029),
        ]
    )

    img = transform(Image.open(img_path).convert("L")).unsqueeze(0)
    x = img.to(device, dtype=torch.float32)

    """ MODEL INIT """
    if model_type == "ResNet-50":
        model_path = os.path.join("models", f"{model_type}")
        model = RoadSegNN(backbone_type=model_type)

    elif model_type == "ResNet-101":
        # model_path = f"models\\{model_type}"
        model_path = os.path.join("models", f"{model_type}")
        model = RoadSegNN(backbone_type=model_type)

    elif model_type == "Swin-T":
        # model_path = f"models\\{model_type}"
        model_path = os.path.join("models", f"{model_type}")
        model = RoadSegNN(backbone_type=model_type)

    elif model_type == "SegNet":
        # model_path = f"models\\{model_type}"
        model_path = os.path.join("models", f"{model_type}")
        model = SegNet()

    model.load_weights(os.path.join(model_path, "ckpt.pth"))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        y = model(x)
        y = (y >= 0.5) * 1.0

    # Sat image reopened for viewing
    x2 = np.array(Image.open(img_path).convert("L"))
    y = F.interpolate(y, (x2.shape[0], x2.shape[1]))
    y = y[0].permute(1, 2, 0).cpu().numpy()

    plt.figure()
    plt.imshow(x2, cmap="gray")
    plt.imshow(y, alpha=0.5)
    plt.axis("off")

    output_path = os.path.abspath(os.path.join(os.getcwd(), "static", "output.png"))
    print(output_path)

    # Save the plot as an image
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    output_path = os.path.join("static", "output.png")
    output_path = output_path.replace("\\", "\\\\")
    # output_path = output_path.replace(')', '\)')

    print(f"Plot saved to {output_path}")
    return output_path
