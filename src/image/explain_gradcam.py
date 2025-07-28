# src/image/explain_gradcam.py

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for the input image.

        Args:
            input_tensor (Tensor): Preprocessed input image of shape [1, C, H, W]
            class_idx (int, optional): Class index to compute Grad-CAM for. If None, uses predicted class.

        Returns:
            heatmap (np.array): Grad-CAM heatmap (H x W)
            predicted_class (int): Predicted class index
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output).item()

        target = output[0, class_idx]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global average pooling
        weighted_activations = weights * self.activations
        cam = weighted_activations.sum(dim=1).squeeze().cpu().numpy()

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam -= cam.min()
        cam /= cam.max()
        return cam, class_idx

def overlay_heatmap_on_image(image_pil, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image_pil (PIL.Image): Original image
        heatmap (np.array): Grad-CAM heatmap
        alpha (float): Transparency
        colormap: OpenCV colormap

    Returns:
        Image with heatmap overlay (PIL.Image)
    """
    image_np = np.array(image_pil.resize(heatmap.shape[::-1]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    overlayed = cv2.addWeighted(heatmap_color, alpha, image_np, 1 - alpha, 0)
    return Image.fromarray(overlayed)
