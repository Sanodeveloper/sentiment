import tensorflow as tf
from transformers import BertTokenizer
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.utils import shuffle  # データシャッフル用

def train_sentiment_model(input_csv, model_save_path):
    """
    Trains a sentiment analysis model using the given CSV file and saves it in TensorFlow format.

    Args:
        input_csv (str): Path to the input CSV file.
        model_save_path (str): Path to save the TensorFlow model.
    """
    # Load the dataset
    data = pd.read_csv(input_csv)

    data = shuffle(data, random_state=42)

    # Prepare the text and labels
    sentences = data['sentence'].values
    labels = data['sentiment'].values

    # Load the BERT tokenizer
    model_name = "cl-tohoku/bert-base-japanese"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize the text using BERT tokenizer
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

    # Split the data into training and validation sets
    train_size = int(len(sentences) * 0.8)
    train_input_ids = input_ids[:train_size]
    train_attention_masks = attention_masks[:train_size]
    train_labels = labels[:train_size]
    val_input_ids = input_ids[train_size:]
    val_attention_masks = attention_masks[train_size:]
    val_labels = labels[train_size:]

    # Build the model
    input_ids_layer = Input(shape=(100,), dtype=tf.int32, name="input_ids")
    attention_masks_layer = Input(shape=(100,), dtype=tf.int32, name="attention_mask")

    # Embedding layer (trainable)
    embedding_layer = tf.keras.layers.Embedding(input_dim=30522, output_dim=128, input_length=100)(input_ids_layer)

    # Global Average Pooling
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)

    # Fully connected layers
    dense = Dense(64, activation='relu')(pooled_output)
    dropout = Dropout(0.3)(dense)
    output = Dense(3, activation='softmax')(dropout)

    model = Model(inputs=[input_ids_layer, attention_masks_layer], outputs=output)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  metrics=['accuracy'])

    # Train the model using GPU
    with tf.device('/GPU:0'):  # 明示的にGPUを指定
        model.fit(
            [train_input_ids, train_attention_masks],
            train_labels,
            validation_data=([val_input_ids, val_attention_masks], val_labels),
            epochs=20,
            batch_size=16
        )

    # Save the model in TensorFlow format
    model.save(model_save_path)

# Example usage
input_csv = 'japanese_sentence_with_sentiment.csv'
model_save_path = 'sentiment_model'
train_sentiment_model(input_csv, model_save_path)