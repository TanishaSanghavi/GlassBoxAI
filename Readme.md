

# 🧠 GlassBoxAI

An MultiModal XAI tool to visualize and interpret predictions made by text and image classifiers using LIME, Grad-CAM, and counterfactual explanations.

---

## Features

- **Text Classification**
  - ✅ LIME explanations (highlighted words)
  - ✅ SHAP explanations (highlighted words)
  - ✅ Counterfactuals using antonyms (filtered to impactful POS)
- **Image Classification**
  - ✅ Grad-CAM heatmaps
  - ✅ Visual counterfactual variants

---

## Setup Instructions

1. **Clone & Navigate**
   ```bash
   git clone https://github.com/yourusername/ExplainMyBlackBox.git
   cd ExplainMyBlackBox
    ```
2. **Create Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate
    ```

3. **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Resources**
    ```bash
    python -m nltk.downloader wordnet omw-1.4
    python -m spacy download en_core_web_sm
    ```

5. **Run the App**
    ```bash
    streamlit run ui/streamlit_app.py
    ```

# Python Version
Recommended: Python 3.10

🚀 [Live Demo](https://glassboxai-6gravet6doysudhhkzjxww.streamlit.app/)
