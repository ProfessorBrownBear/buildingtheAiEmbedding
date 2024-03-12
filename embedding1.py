import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
# this image displays what the final embedding looks like
# https://app.screencast.com/h72wvpI5YXUXn
# remember that the embedding is an Algebraic Matrix which numerically encodes the tokens and weightings in the training data set.

# Sample dataset
sentences = [
    'I love machine learning',
    'I love coding in Python',
    'I enjoy deep learning'
]

# Tokenize the sentences
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)

# Padding the sequences
padded = pad_sequences(sequences, padding='post')

# Define the embedding model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=100, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Print the summary of the model
model.summary()

# Training the model with the same data as input and output
model.fit(padded, np.array([1, 1, 1]), epochs=10)

# Get the weights from the embedding layer
e = model.layers[0]
weights = e.get_weights()[0]

# Mapping words to their embeddings
for word, i in word_index.items():
    if i < 100:
        print(word, ": ", weights[i])
