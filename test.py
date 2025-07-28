from src.text.counterfactuals import generate_counterfactuals

text = "The movie was dull and boring"
cf_sentences = generate_counterfactuals(text, num_variants=3)

print("Original:", text)
print("Counterfactuals:")
for sent in cf_sentences:
    print("-", sent)
