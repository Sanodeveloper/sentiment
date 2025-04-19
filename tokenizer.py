import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

sentences = [
    "私は日本が好きです。",  # I like Japan.
    "今日はいい天気ですね。",  # It's nice weather today.
    "あなたはどう思いますか？",  # What do you think?
    "私は元気です。",  # I am fine.
    "明日は雨が降るかもしれません。"  # It might rain tomorrow.
]

#  # Tokenize the text
# tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
# tokenizer.fit_on_texts(sentences)
# sequences = tokenizer.texts_to_sequences(sentences)
# print("Sequences:", sequences)
# print("Word Index:", tokenizer.word_index)
# print("Word Counts:", tokenizer.word_counts)
# padded_sequences = pad_sequences(sequences, padding='post', maxlen=100)