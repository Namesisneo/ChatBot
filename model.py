import spacy
import json
from spacy.training import Example
import random


# Load JSON training data
with open('train_data.json', 'r') as f:
    raw_train_data = json.load(f)

# Initialize spaCy blank English model
nlp = spacy.blank("en")

# Add text classification pipeline
if "textcat" not in nlp.pipe_names:
    text_classifier = nlp.add_pipe("textcat", last=True)
else:
    text_classifier = nlp.get_pipe("textcat")

# Add labels dynamically from the data
labels = set()
for text, annotations in raw_train_data:
    labels.update(annotations["cats"].keys())

for label in labels:
    text_classifier.add_label(label)

# Convert raw data to spaCy training examples
train_examples = []
for text, annotations in raw_train_data:
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    train_examples.append(example)

# Begin training
optimizer = nlp.begin_training()
epochs = 20
for epoch in range(epochs):
    random.shuffle(train_examples)
    losses = {}
    for example in train_examples:
        nlp.update([example], drop=0.3, losses=losses)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {losses['textcat']:.4f}")

# Save trained model
nlp.to_disk("intent_model")

# Test the model
test_text = "What is the admission process for engineering colleges?"
model = spacy.load("intent_model")
doc = model(test_text)

# Print detected intents
print("\nDetected Intents:")
for label, score in doc.cats.items():
    if score > 0.5:  # Confidence threshold
        print(f"Intent: {label}, Confidence: {score:.2f}")

# print(spacy.prefer_gpu())
