# src/load_model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model(model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
    """
    Loads a pre-trained text classification model and its tokenizer.

    Args:
        model_name (str): HuggingFace model name.

    Returns:
        tokenizer, model (on CPU or GPU if available)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set to inference mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device
