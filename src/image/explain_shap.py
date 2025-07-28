# src/image/explain_shap.py

import shap
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def explain_with_shap(image, model, preprocess, device, class_names, background_images=None):
    """
    Generates SHAP explanation for an image classification model.

    Args:
        image (PIL.Image): Input image to explain.
        model (torch.nn.Module): Pretrained image classification model.
        preprocess: Preprocessing pipeline.
        device: 'cuda' or 'cpu'.
        class_names (list): Class names corresponding to model outputs.
        background_images (list[PIL.Image], optional): Background dataset for SHAP (defaults to 5 copies of the input).

    Returns:
        Tuple: (shap_values_np, predicted_label, class_name)
    """

    # Preprocess single image
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_pred_idx = torch.argmax(probs).item()
        predicted_label = top_pred_idx
        class_name = class_names[top_pred_idx] if class_names else str(top_pred_idx)

    # Define SHAP prediction function
    def shap_predict(images_np):
        model.eval()
        imgs_tensor = torch.stack([
            preprocess(Image.fromarray((img * 255).astype(np.uint8))).to(device)
            for img in images_np
        ])
        with torch.no_grad():
            output = model(imgs_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
        return probs.cpu().numpy()

    # Background dataset
    if background_images is None:
        background_images = [image] * 5  # use 5 duplicates of input
    background_np = np.stack([
        np.array(img.resize(image.size)).astype(np.float32) / 255.0
        for img in background_images
    ])

    # SHAP image explainer
    explainer = shap.Explainer(shap_predict, background_np, algorithm="permutation")
    shap_values = explainer(np.array(image).astype(np.float32)[None, ...] / 255.0)

    # Convert SHAP values to numpy for visualization
    shap_values_np = shap_values.values[0]

    return shap_values_np, predicted_label, class_name
