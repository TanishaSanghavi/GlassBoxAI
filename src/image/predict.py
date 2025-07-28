# src/image/predict.py

import torch
from PIL import Image
import os

def predict_image(image_input, model, preprocess, class_names, device, topk=5):
    """
    Predict the top classes for a given image.

    Args:
        image_input (str or PIL.Image): Path to image or PIL image
        model (torch.nn.Module): Pretrained image model
        preprocess (torchvision.transforms): Transform pipeline
        class_names (List[str]): Human-readable class labels
        device (torch.device): Device to run model on
        topk (int): Number of top predictions to return

    Returns:
        List[Tuple[str, float]]: Top K class labels and probabilities
    """
    
    # Load image if given as a path
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("Invalid image_input. Provide image path or PIL.Image.")

    # Apply preprocessing
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get top predictions
    top_probs, top_indices = torch.topk(probabilities, topk)
    results = [(class_names[idx], round(prob.item(), 4)) for idx, prob in zip(top_indices, top_probs)]

    return results
