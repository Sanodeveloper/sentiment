import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

def train_sentiment_model(input_csv, model_save_path):
    """
    Trains a sentiment analysis model using the given CSV file and saves it in TFLite format.

    Args:
        input_csv (str): Path to the input CSV file.
        model_save_path (str): Path to save the TFLite model.
    """
    # Load the dataset
    data = pd.read_csv(input_csv)

    # Prepare the text and labels
    sentences = data['sentence'].values
    labels = data['sentiment'].values

    # Tokenize the text
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)

    # Split the data into training and validation sets
    train_size = int(len(sentences) * 0.8)
    train_sequences = padded_sequences[:train_size]
    train_labels = labels[:train_size]
    val_sequences = padded_sequences[train_size:]
    val_labels = labels[train_size:]

    # Build the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000, 16, input_length=100),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_sequences, train_labels, epochs=10, validation_data=(val_sequences, val_labels))

    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(model_save_path, 'wb') as f:
        f.write(tflite_model)

# Example usage
input_csv = 'japanese_sentence_with_sentiment.csv'
model_save_path = 'sentiment_model.tflite'
train_sentiment_model(input_csv, model_save_path)