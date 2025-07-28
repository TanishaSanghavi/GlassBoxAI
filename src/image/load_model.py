# src/image/load_model.py

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import json
import os

def load_image_model(model_name="resnet18", use_cuda=True):
    """
    Load a pretrained image classification model from torchvision.
    Args:
        model_name (str): Name of the model (e.g., 'resnet18', 'vgg16')
        use_cuda (bool): Whether to use GPU if available
    Returns:
        model (torch.nn.Module): Loaded and ready-to-use model
        preprocess (torchvision.transforms): Image preprocessing pipeline
        class_names (List[str]): List of class labels
    """

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # Load pretrained model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    model.to(device)

    # Define standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load class labels (ImageNet)
    class_names = load_imagenet_class_names()

    return model, preprocess, class_names, device


def load_imagenet_class_names():
    """
    Load human-readable ImageNet class labels (from local json or URL).
    Returns:
        List[str]: 1000 class names
    """
    json_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.json")

    # If not available locally, download from torchvision
    if not os.path.exists(json_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, json_path.replace(".json", ".txt"))

        # Convert to JSON format
        with open(json_path.replace(".json", ".txt")) as f:
            classes = [line.strip() for line in f.readlines()]
        with open(json_path, "w") as jf:
            json.dump(classes, jf)
    else:
        with open(json_path, "r") as jf:
            classes = json.load(jf)

    return classes
