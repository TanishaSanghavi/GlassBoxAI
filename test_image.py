import argparse
from PIL import Image
import torch

from src.image.load_model import load_image_model
from src.image.predict import predict_image
from src.image.explain_gradcam import GradCAM
from src.image.explain_lime import explain_with_lime
from src.image.explain_shap import explain_with_shap
from src.image.utils import preprocess_image, load_imagenet_labels, decode_prediction, tensor_to_pil

def main(args):
    # Load model and labels
    model = load_image_model(args.model_name)
    idx_to_label = load_imagenet_labels(args.labels_json)

    # Load and preprocess image
    img = Image.open(args.image_path).convert("RGB")
    input_tensor = preprocess_image(img)

    # Predict class
    output = predict_image(model, input_tensor)
    label, confidence = decode_prediction(output, idx_to_label)
    print(f"\nPredicted Class: {label} (Confidence: {confidence:.2f})\n")

    # Visual Explanations
    if args.gradcam:
        print("[*] Generating Grad-CAM heatmap...")
        gradcam_path = generate_gradcam(model, input_tensor, args.image_path)
        print(f"✔ Grad-CAM saved to: {gradcam_path}")

    if args.lime:
        print("[*] Generating LIME explanation...")
        lime_path = generate_lime_explanation(model, args.image_path, args.model_name)
        print(f"✔ LIME result saved to: {lime_path}")

    if args.shap:
        print("[*] Generating SHAP explanation...")
        shap_path = generate_shap_explanation(model, input_tensor)
        print(f"✔ SHAP result saved to: {shap_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model to load")
    parser.add_argument("--labels_json", type=str, default=None, help="Path to label mapping (optional)")

    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM explanation")
    parser.add_argument("--lime", action="store_true", help="Generate LIME explanation")
    parser.add_argument("--shap", action="store_true", help="Generate SHAP explanation")

    args = parser.parse_args()
    main(args)
