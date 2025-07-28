# src/image/explain_lime.py

import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch
from PIL import Image  # Make sure this is imported

def explain_with_lime(image, model, preprocess, device, class_names, top_label_idx=None):
    """
    Generates LIME explanation for an image.

    Args:
        image (PIL.Image): The image to explain.
        model (torch.nn.Module): The trained image classification model.
        preprocess: Image preprocessing function.
        device: CPU/GPU.
        class_names: List of class labels.
        top_label_idx (int, optional): Index of the class to explain (default: predicted class)

    Returns:
        Tuple: (explanation image, explanation text, heatmap mask)
    """

    # LIME expects numpy image in HWC format
    np_img = np.array(image)

    # Prepare LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Define prediction function for LIME
    def batch_predict(images):
        model.eval()
        batch = torch.stack([
            preprocess(Image.fromarray(img)).to(device)
            for img in images
        ])
        with torch.no_grad():
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()

    # Run LIME (faster config)
    explanation = explainer.explain_instance(
        np_img,
        batch_predict,
        top_labels=1,       # Only explain top-1 class
        hide_color=0,
        num_samples=50      # Reduced number of perturbations (was 100+)
    )

    label_to_explain = top_label_idx or explanation.top_labels[0]

    temp, mask = explanation.get_image_and_mask(
        label=label_to_explain,
        positive_only=True,
        num_features=5,     # fewer features for speed
        hide_rest=False
    )

    highlighted_image = mark_boundaries(temp / 255.0, mask)
    label_text = class_names[label_to_explain] if class_names else str(label_to_explain)

    return highlighted_image, label_text, mask
