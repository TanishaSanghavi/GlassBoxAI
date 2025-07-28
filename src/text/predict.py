# src/predict.py

import torch
import torch.nn.functional as F
from typing import Tuple, List

def predict(text: str, tokenizer, model, device, label_map=None) -> Tuple[str, List[float]]:
    """
    Predicts the label and confidence scores for input text.

    Args:
        text (str): Input sentence.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model.
        device: CPU or GPU.
        label_map (dict, optional): Maps class index to label names.

    Returns:
        predicted_label (str), class_probabilities (List[float])
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().tolist()

    pred_idx = torch.argmax(logits, dim=1).item()

    if label_map:
        pred_label = label_map[pred_idx]
    else:
        pred_label = str(pred_idx)

    return pred_label, probs
