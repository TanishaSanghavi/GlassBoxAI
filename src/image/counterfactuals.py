# src/image/counterfactuals.py

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def generate_adversarial_example(model, input_tensor, epsilon=0.01, target_class=None):
    """
    Generate a simple counterfactual image using FGSM (Fast Gradient Sign Method).

    Args:
        model (torch.nn.Module): Pretrained image classification model.
        input_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W].
        epsilon (float): Perturbation strength.
        target_class (int, optional): If specified, generate targeted adversarial example.

    Returns:
        adv_tensor (torch.Tensor): Perturbed image tensor.
        original_pred (int): Original prediction class.
        adv_pred (int): Adversarial prediction class.
    """
    model.eval()
    input_tensor.requires_grad = True

    output = model(input_tensor)
    original_pred = torch.argmax(output, dim=1).item()

    if target_class is None:
        target_class = (original_pred + 1) % output.shape[1]  # Pick a different class

    loss = F.cross_entropy(output, torch.tensor([target_class]))
    model.zero_grad()
    loss.backward()

    perturbation = epsilon * input_tensor.grad.sign()
    adv_tensor = input_tensor + perturbation
    adv_tensor = torch.clamp(adv_tensor, 0, 1)  # Ensure valid pixel range

    adv_output = model(adv_tensor)
    adv_pred = torch.argmax(adv_output, dim=1).item()

    return adv_tensor.detach(), original_pred, adv_pred

def tensor_to_pil(tensor):
    """
    Convert a normalized torch tensor to a PIL image.

    Args:
        tensor (torch.Tensor): Tensor of shape [1, C, H, W]

    Returns:
        PIL.Image
    """
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    image = unnormalize(tensor.squeeze(0)).clamp(0, 1)
    image_np = image.permute(1, 2, 0).detach().cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)
