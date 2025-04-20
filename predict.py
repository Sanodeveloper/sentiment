import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np

def load_model_and_predict(model_path, sentences, model_name="cl-tohoku/bert-base-japanese"):
    """
    Loads a saved TensorFlow model and performs sentiment prediction on given sentences.

    Args:
        model_path (str): Path to the saved TensorFlow model.
        sentences (list): List of sentences to predict.
        model_name (str): Name of the BERT model used for tokenization.

    Returns:
        list: Predicted sentiment labels for the input sentences.
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path, custom_objects={"TFBertModel": TFBertModel})

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize the input sentences
    def tokenize_sentences(sentences):
        input_ids = []
        attention_masks = []
        for sentence in sentences:
            encoded = tokenizer.encode_plus(
                sentence,
                max_length=100,
                padding="max_length",
                truncation=True,
                return_tensors="tf"
            )
            input_ids.append(encoded["input_ids"])
            attention_masks.append(encoded["attention_mask"])
        return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)

    input_ids, attention_masks = tokenize_sentences(sentences)

    # Perform prediction
    predictions = model.predict([input_ids, attention_masks])
    predicted_labels = np.argmax(predictions, axis=1)  # Get the index of the highest probability

    return predicted_labels

# Example usage
if __name__ == "__main__":
    model_path = "sentiment_model.h5"  # Path to the saved model
    sentences = [
        "このまま終わらせるなんて、嫌だった"
    ]

    predicted_labels = load_model_and_predict(model_path, sentences)
    labels = {0: "Positive", 1: "Neutral", 2: "Negative"}
    for sentence, label in zip(sentences, predicted_labels):
        print(f"{sentence}: {labels[label]}")