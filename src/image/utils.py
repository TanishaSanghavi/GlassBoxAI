import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os

# -----------------------------------
# Image Preprocessing
# -----------------------------------

def get_image_transform():
    """Return standard image transforms for pretrained models (e.g., ResNet)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet means
            std=[0.229, 0.224, 0.225]     # ImageNet stds
        )
    ])

def preprocess_image(img: Image.Image):
    """Preprocess a PIL image for model input."""
    transform = get_image_transform()
    return transform(img).unsqueeze(0)  # Add batch dimension

# -----------------------------------
# Post-processing and Labels
# -----------------------------------

def load_imagenet_labels(json_path=None):
    """Load class index to label mapping (e.g., from ImageNet)."""
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r') as f:
            idx_to_label = json.load(f)
    else:
        # Default to dummy labels (0-999) if no mapping is provided
        idx_to_label = {str(i): f"class_{i}" for i in range(1000)}
    return idx_to_label

def decode_prediction(output_tensor, idx_to_label):
    """Convert output tensor to class label and confidence."""
    probs = torch.nn.functional.softmax(output_tensor, dim=1)[0]
    conf, pred_idx = torch.max(probs, dim=0)
    label = idx_to_label.get(str(pred_idx.item()), f"class_{pred_idx.item()}")
    return label, conf.item()

# -----------------------------------
# Visualization Helpers
# -----------------------------------

def tensor_to_pil(tensor):
    """Convert tensor image (1, 3, H, W) to PIL for display."""
    tensor = tensor.squeeze(0)  # Remove batch dimension
    inv_transform = transforms.Compose([
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.Normalize(
            mean=[-0.485, -0.456, -0.406],
            std=[1., 1., 1.]
        ),
        transforms.ToPILImage()
    ])
    return inv_transform(tensor.cpu().clone())

def save_tensor_as_image(tensor, filename):
    """Save tensor as image file."""
    img = tensor_to_pil(tensor)
    img.save(filename)

def preprocess_image_upload(uploaded_file, preprocess, device):
    """
    Preprocess an uploaded image file (e.g., from Streamlit) for model input.

    Args:
        uploaded_file: Streamlit's uploaded file object.
        preprocess: torchvision transform function.
        device: torch device.

    Returns:
        Preprocessed image tensor on the given device.
    """
    image = Image.open(uploaded_file).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor
