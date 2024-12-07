import spacy
import json
import os

class IntentClassifier:
    def __init__(self, model_path='intent_model'):
        """
        Initialize the intent classifier
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = model_path
        self.nlp = None
    
    def train_model(self, train_data_path, force_retrain=False):
        """
        Train the model if not already trained
        
        Args:
            train_data_path (str): Path to training data JSON
            force_retrain (bool): Force retraining even if model exists
        """
        # Check if model already exists and we're not force retraining
        if os.path.exists(self.model_path) and not force_retrain:
            print("Model already exists. Loading existing model.")
            self.nlp = spacy.load(self.model_path)
            return
        
        # Load training data
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Create blank English model
        nlp = spacy.blank("en")
        text_classifier = nlp.add_pipe("textcat")
        
        # Collect all categories
        categories = set()
        for _, annotations in train_data:
            categories.update(annotations['cats'].keys())
        
        # Add categories to classifier
        for category in categories:
            text_classifier.add_label(category)
        
        # Prepare training examples
        train_examples = []
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = spacy.training.Example.from_dict(
                doc, 
                {"cats": annotations['cats']}
            )
            train_examples.append(example)
        
        # Initialize and train
        optimizer = nlp.begin_training()
        for _ in range(10):  # 10 epochs
            losses = {}
            nlp.update(train_examples, sgd=optimizer, losses=losses)
        
        # Save the trained model
        nlp.to_disk(self.model_path)
        print(f"Model trained and saved to {self.model_path}")
        
        self.nlp = nlp
    
    def predict(self, texts):
        """
        Predict intent categories for given texts
        
        Args:
            texts (list or str): Text or list of texts to classify
        
        Returns:
            list: Predictions for each text
        """
        if self.nlp is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        predictions = []
        for text in texts:
            doc = self.nlp(text)
            # Sort categories by confidence score
            sorted_cats = sorted(
                doc.cats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            predictions.append({
                'text': text,
                'categories': dict(sorted_cats)
            })
        
        return predictions

# Example usage
def main():
    # Initialize classifier
    classifier = IntentClassifier()
    
    # Train the model (will only train if model doesn't exist)
    classifier.train_model('train_data.json')
    
    # Test predictions
    test_texts = [
        "What is the admission process for engineering colleges?",
        "Tell me about scholarship opportunities",
        "What are the hostel facilities?",
    ]
    
    predictions = classifier.predict(test_texts)
    
    # Print predictions
    for pred in predictions:
        print(f"\nText: {pred['text']}")
        print("Top Categories:")
        for cat, score in pred['categories'].items():
            if score > 0.1:  # Only show categories with >10% confidence
                print(f"{cat}: {score:.2f}")

if __name__ == "__main__":
    main()