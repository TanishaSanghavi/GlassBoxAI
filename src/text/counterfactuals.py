import random
from typing import List
from nltk.corpus import wordnet
import spacy
import os

def get_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to local relative model
        model_path = os.path.join(os.path.dirname(__file__), "en_core_web_sm")
        return spacy.load(model_path)

nlp = get_nlp()


# ---------- Utility Functions ----------

def get_antonyms(word: str) -> List[str]:
    antonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                antonym = ant.name().replace("_", " ")
                if antonym.lower() != word.lower():
                    antonyms.add(antonym)
    return list(antonyms)

def get_synonyms(word: str) -> List[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                synonyms.add(name)
    return list(synonyms)

# ---------- Main Counterfactual Generator ----------

def generate_counterfactuals(text: str, num_variants: int = 3) -> List[str]:
    doc = nlp(text)
    tokens = [token.text for token in doc]

    counterfactuals = set()

    for i, token in enumerate(doc):
        pos = token.pos_
        if pos in {"ADJ", "ADV", "VERB"}:
            # Try antonyms first
            antonyms = get_antonyms(token.text)
            random.shuffle(antonyms)

            replaced = False
            for antonym in antonyms:
                new_tokens = tokens[:i] + [antonym] + tokens[i+1:]
                new_sentence = " ".join(new_tokens)
                if new_sentence.lower() != text.lower():
                    counterfactuals.add(new_sentence)
                    replaced = True
                    break

            # Fallback: Try synonyms only if antonym didn't work
            if not replaced:
                synonyms = get_synonyms(token.text)
                random.shuffle(synonyms)
                for synonym in synonyms:
                    new_tokens = tokens[:i] + [synonym] + tokens[i+1:]
                    new_sentence = " ".join(new_tokens)
                    if new_sentence.lower() != text.lower():
                        counterfactuals.add(new_sentence)
                        break

        if len(counterfactuals) >= num_variants:
            break

    return list(counterfactuals)
