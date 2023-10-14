import numpy as np
import tensorflow as tf
from keras import layers, Sequential
import training_data_generator

tagged_sequences = training_data_generator.generate_training_data()



def train_and_save_model(sequences, labels_encoded, unique_labels, epochs, batch_size):
# Define the LSTM model
    model = Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(75, 6)), # input shape is (sequence length * 25, 6) 
        layers.LSTM(64),
        layers.Dense(len(unique_labels), activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #  training model
    model.fit(sequences, labels_encoded, epochs, batch_size)
    model.save("test_model.keras")
