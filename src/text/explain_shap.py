# src/explain_shap.py

import shap
import torch
import numpy as np
from typing import Tuple, List


class ShapExplainer:
    def __init__(self, tokenizer, model, device, label_names: List[str]):
        """
        SHAP explainer for Transformer-based classifiers.
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.label_names = label_names

        self.explainer = shap.Explainer(self._predict_proba, shap.maskers.Text(tokenizer))

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Custom wrapper for HuggingFace models to return prediction probabilities.
        Used internally by SHAP.
        """
        self.model.eval()
        all_probs = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs[0])

        return np.array(all_probs)

    def explain(self, text: str, class_idx: int = None, num_features: int = 10) -> Tuple[str, List[Tuple[str, float]]]:
        """
        Generate SHAP explanation for the given input text.

        Returns:
            predicted_label (str),
            List of (token, shap_value) for the predicted class
        """
        shap_values = self.explainer([text])
        pred_idx = np.argmax(shap_values[0].values.sum(axis=1))

        if class_idx is None:
            class_idx = pred_idx

        # Defensive: handle unknown label index
        if class_idx >= len(self.label_names):
            label = f"Class {class_idx}"
        else:
            label = self.label_names[class_idx]

        tokens = shap_values[0].data
        values = shap_values[0].values[class_idx]

        token_scores = list(zip(tokens, values))
        token_scores.sort(key=lambda x: abs(x[1]), reverse=True)

        # ✅ Clean tokens to match input text (remove ▁, strip)
        cleaned_token_scores = [
            (token.lstrip("▁").strip(), score)
            for token, score in token_scores
            if token.strip()
        ]

        return label, cleaned_token_scores[:num_features]
