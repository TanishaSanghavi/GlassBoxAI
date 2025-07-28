import matplotlib.pyplot as plt
from typing import List, Tuple
import string


def plot_confidence_bar(labels: List[str], probs: List[float]):
    fig, ax = plt.subplots()
    bars = ax.bar(labels, probs, color="skyblue")
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{prob:.2f}", ha='center', fontsize=9)

    return fig


def highlight_text(text: str, explanation: List[Tuple[str, float]], threshold: float = 0.01) -> str:
    """
    Highlights words based on their importance score.
    Positive → green, Negative → red
    """

    from html import escape

    # Normalize explanation tokens
    explanation_dict = {
        token.strip(string.punctuation).lower(): score
        for token, score in explanation
    }

    tokens = text.split()
    html = []

    for token in tokens:
        clean_token = token.strip(string.punctuation).lower()
        score = explanation_dict.get(clean_token, 0.0)

        if abs(score) >= threshold:
            color = "rgba(0, 200, 0, 0.4)" if score > 0 else "rgba(255, 0, 0, 0.4)"
            html.append(f"<span style='background-color:{color}; padding:2px'>{escape(token)}</span>")
        else:
            html.append(escape(token))

    return " ".join(html)
