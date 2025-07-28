# src/explain_lime.py

from lime.lime_text import LimeTextExplainer
import numpy as np
from typing import List, Tuple

class LimeExplainer:
    def __init__(self, tokenizer, model, device, label_names: List[str]):
        """
        Initialize the LIME explainer.

        Args:
            tokenizer: HuggingFace tokenizer
            model: HuggingFace model
            device: torch.device("cpu" or "cuda")
            label_names: List of label names (index = class ID)
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.label_names = label_names
        self.explainer = LimeTextExplainer(class_names=label_names)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for a list of texts — LIME will call this internally.

        Args:
            texts (List[str])

        Returns:
            probs (np.ndarray): Shape = [n_samples, n_classes]
        """
        import torch
        import torch.nn.functional as F

        all_probs = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs[0])
        return np.array(all_probs)

    def explain(self, text: str, num_features: int = 10) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Generates a LIME explanation for the input text.

        Args:
            text (str): Input string
            num_features (int): How many important words to show

        Returns:
            predicted_label (str)
            explanation: List of (word, weight) tuples
        """
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=self.predict_proba,
            num_features=num_features
        )

        # Safe prediction label inference
        pred_label_idx = max(exp.available_labels(), key=lambda l: exp.predict_proba[l])
        label = self.label_names[pred_label_idx]

        return label, exp.as_list(label=pred_label_idx)
