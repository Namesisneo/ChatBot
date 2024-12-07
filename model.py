import json
import random
import numpy as np
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example

class AdvancedTextClassifier:
    def __init__(self, train_data_path):
        """
        Initialize text classifier with robust data loading and preprocessing
        
        Args:
            train_data_path (str): Path to training data JSON file
        """
        # Load and preprocess data
        self.train_data, self.val_data, self.categories = self.load_and_validate_data(train_data_path)
        
        # Initialize spaCy model
        self.nlp = self._initialize_model()
        
        # Prepare training examples
        self.train_examples = self._prepare_spacy_examples(self.train_data)
        self.val_examples = self._prepare_spacy_examples(self.val_data)
    
    def load_and_validate_data(self, file_path):
        """
        Load and validate training data from JSON file
        
        Returns:
            tuple: Processed train and validation datasets with categories
        """
        with open(file_path, "r", encoding='utf-8') as f:
            train_data = json.load(f)
        
        # Print dataset information
        print("Dataset Information:")
        print(f"Total number of examples: {len(train_data)}")
        
        # Collect all unique categories
        all_categories = sorted(set(
            category 
            for _, annotations in train_data 
            for category in annotations['cats'].keys()
        ))
        print("Categories found:", all_categories)
        
        # Normalize data to ensure consistent category representation
        normalized_data = []
        for text, annotations in train_data:
            # Create a complete category dictionary
            complete_cats = {
                cat: float(annotations['cats'].get(cat, 0.0)) 
                for cat in all_categories
            }
            normalized_data.append([text, {'cats': complete_cats}])
        
        # Shuffle and split data
        random.shuffle(normalized_data)
        train_ratio = 0.8
        train_size = int(len(normalized_data) * train_ratio)
        
        train_data_list = normalized_data[:train_size]
        val_data_list = normalized_data[train_size:]
        
        print(f"Training set size: {len(train_data_list)}")
        print(f"Validation set size: {len(val_data_list)}")
        
        return train_data_list, val_data_list, all_categories
    
    def _initialize_model(self):
        """
        Initialize spaCy model with text categorizer
        
        Returns:
            spacy.Language: Configured spaCy language model
        """
        nlp = spacy.blank("en")
        text_classifier = nlp.add_pipe("textcat")
        
        # Add labels to the text classifier
        for category in self.categories:
            text_classifier.add_label(category)
        
        # Reinitialize the model after adding labels
        nlp.initialize()
        
        return nlp
    
    def _prepare_spacy_examples(self, data_list):
        """
        Convert data to spaCy examples
        
        Args:
            data_list (list): List of training/validation data
        
        Returns:
            list: List of spaCy training examples
        """
        examples = []
        for text, annotations in data_list:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, {"cats": annotations["cats"]})
            examples.append(example)
        return examples
    
    def custom_validation_loss(self):
        """
        Calculate custom multi-label validation loss
        
        Returns:
            float: Validation loss
        """
        total_loss = 0.0
        
        for example in self.val_examples:
            # Ensure predictions are made
            doc = self.nlp(example.reference.text)
            
            # Extract true labels and predicted scores
            true_labels = list(example.reference.cats.values())
            pred_scores = list(doc.cats.values())
            
            # Ensure we have predictions for all categories
            if not pred_scores:
                print(f"Warning: No predictions for text: {example.reference.text}")
                continue
            
            # Clip scores to prevent log(0)
            pred_scores = np.clip(pred_scores, 1e-15, 1 - 1e-15)
            
            # Binary cross-entropy loss
            bce_loss = -(
                np.array(true_labels) * np.log(pred_scores) + 
                (1 - np.array(true_labels)) * np.log(1 - pred_scores)
            )
            
            total_loss += np.mean(bce_loss)
        
        return total_loss / len(self.val_examples)
    
    def train(self, epochs=30, patience=5):
        """
        Train the text classifier with early stopping
        
        Args:
            epochs (int): Maximum number of training epochs
            patience (int): Number of epochs to wait for improvement
        """
        optimizer = self.nlp.begin_training()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle training examples
            random.shuffle(self.train_examples)
            losses = {}
            
            # Create minibatches
            batches = minibatch(self.train_examples, size=compounding(4.0, 32.0, 1.001))
            
            # Update model
            for batch in batches:
                self.nlp.update(batch, sgd=optimizer, losses=losses, drop=0.3)
            
            # Calculate validation loss
            val_loss = self.custom_validation_loss()
            
            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"Training Loss: {losses.get('textcat', 0.0):.4f}, "
                f"Validation Loss: {val_loss:.4f}"
            )
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.nlp.to_disk("best_intent_model")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        
        # Restore best model
        self.nlp = spacy.load("best_intent_model")
    
    def predict(self, texts):
        """
        Predict categories for given texts
        
        Args:
            texts (list): List of texts to predict
        
        Returns:
            list: Predictions for each text
        """
        predictions = []
        for text in texts:
            doc = self.nlp(text)
            predictions.append({
                'text': text,
                'categories': doc.cats
            })
        return predictions

# Usage
def main():
    classifier = AdvancedTextClassifier("train_data.json")
    classifier.train()
    
    # Test predictions
    test_texts = [
        "What is the admission process for engineering colleges?",
        "What are the eligibility criteria for scholarships?",
        "Tell me about hostel facilities"
    ]
    
    predictions = classifier.predict(test_texts)
    for pred in predictions:
        print(f"\nText: {pred['text']}")
        print("Predicted Categories:")
        for cat, score in pred['categories'].items():
            print(f"{cat}: {score:.4f}")

if __name__ == "__main__":
    main()