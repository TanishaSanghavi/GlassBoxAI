import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- TEXT MODULE IMPORTS ---
from src.text.load_model import load_model as load_text_model
from src.text.predict import predict
from src.text.explain_lime import LimeExplainer
from src.text.explain_shap import ShapExplainer
from src.text.counterfactuals import generate_counterfactuals, get_antonyms, get_synonyms
from src.text.utils import plot_confidence_bar, highlight_text

# --- IMAGE MODULE IMPORTS ---
from src.image.load_model import load_image_model
from src.image.predict import predict_image
from src.image.explain_gradcam import GradCAM, overlay_heatmap_on_image
from src.image.utils import preprocess_image, tensor_to_pil
from src.image.counterfactuals import generate_adversarial_example, tensor_to_pil

# --- NLTK DOWNLOAD ---
import nltk
nltk.data.path.append("C:/Users/Administrator/nltk_data")
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# --- CONFIG ---
st.set_page_config(page_title="GlassBoxAI", layout="centered")
st.title("🧠 GlassBoxAI")
st.markdown("Unified interpretability for **text and image models** using LIME, SHAP, GradCAM, and Counterfactuals.")

# --- MODE SELECTION ---
mode = st.radio("Choose Input Type:", ["Text", "Image"])

# --- TEXT MODEL LOAD ---
@st.cache_resource
def get_text_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer, model, device = load_text_model(model_name)
    return tokenizer, model, device

tokenizer, text_model, text_device = get_text_model()
LABELS = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
lime_text = LimeExplainer(tokenizer, text_model, text_device, LABELS)
shap_text = ShapExplainer(tokenizer, text_model, text_device, LABELS)

# --- IMAGE MODEL LOAD ---
@st.cache_resource
def get_image_model():
    return load_image_model(model_name="resnet18", use_cuda=True)

image_model, image_preprocess, class_names, image_device = get_image_model()

# =================== TEXT EXPLANATION =================== #
if mode == "Text":
    st.header("✍️ Text Explanation")

    text = st.text_area("Enter a sentence:", "The plot was weak but the cinematography was stunning.", height=120)
    col1, col2 = st.columns(2)
    with col1:
        method_text = st.selectbox("Text Explanation Method", ["LIME", "SHAP"])
    with col2:
        num_features = st.slider("Top Words", 5, 20, 10)

    if st.button("🔍 Analyze Text"):
        label_map = {i: l for i, l in enumerate(LABELS)}
        pred_label, probs = predict(text, tokenizer, text_model, text_device, label_map)
        st.success(f"**Predicted Label:** `{pred_label}`")
        st.subheader("📊 Confidence Scores")
        st.pyplot(plot_confidence_bar(LABELS, probs))

        if method_text == "LIME":
            _, explanation = lime_text.explain(text, num_features=num_features)
        else:
            _, explanation = shap_text.explain(text, num_features=num_features)

        st.subheader(f"🧠 {method_text} Explanation")
        st.markdown(highlight_text(text, explanation), unsafe_allow_html=True)

        st.subheader("🔁 Counterfactuals")
        cf_sentences = generate_counterfactuals(text, num_variants=3)
        if not cf_sentences:
            st.info("No meaningful counterfactuals generated.")
        else:
            for i, cf_text in enumerate(cf_sentences, 1):
                st.markdown(f"**Variant {i}:** _{cf_text}_")
                cf_label, cf_probs = predict(cf_text, tokenizer, text_model, text_device, label_map)
                st.markdown(f"→ **Predicted Label:** `{cf_label}`")
                st.pyplot(plot_confidence_bar(LABELS, cf_probs))

# =================== IMAGE EXPLANATION =================== #
elif mode == "Image":
    st.header("🖼️ Image Explanation")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_tensor = preprocess_image(image).to(image_device)
        with torch.no_grad():
            output = image_model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = class_names[pred_idx]

        st.success(f"**Predicted Label:** `{pred_label}` ({probs[pred_idx]:.2f})")

        method_img = st.radio("Select Image Explanation Method", ["GradCAM"])

        if method_img == "GradCAM":
            st.subheader("🔍 GradCAM Heatmap")
            gradcam = GradCAM(image_model, target_layer=image_model.layer4[1].conv2)
            heatmap, pred_idx = gradcam.generate_heatmap(img_tensor)
            overlay = overlay_heatmap_on_image(image, heatmap)
            st.image(overlay, caption=f"GradCAM Explanation for '{class_names[pred_idx]}'", use_column_width=True)

            st.markdown(
                "🧠 **GradCAM highlights** the image regions that most influenced the model's prediction. "
                "This is done using the gradients of the last convolutional layer."
            )

        # -------------------- COUNTERFACTUALS --------------------
    st.subheader("🔁 Counterfactual Image (Adversarial Example)")
    if st.button("Generate Counterfactual"):
        adv_tensor, orig_pred_idx, adv_pred_idx = generate_adversarial_example(
            image_model, img_tensor.clone(), epsilon=0.01
        )
        adv_image = tensor_to_pil(adv_tensor)

        st.image(adv_image, caption=f"Counterfactual Prediction: {class_names[adv_pred_idx]}", use_column_width=True)

        # ✅ Fix: use 'image' instead of undefined 'img_pil'
        results = predict_image(image, image_model, image_preprocess, class_names, image_device)
        top_class, top_prob = results[0]

        st.markdown(f"**Original Prediction:** {top_class} ({top_prob * 100:.2f}%)")
        st.markdown(f"**Counterfactual Prediction:** {class_names[adv_pred_idx]}")
