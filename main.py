import numpy as np
import tensorflow as tf
import training_data_generator
from keras import layers, Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split

BATCH_SIZE = 5
EPOCHS = 5
SEQUENCE_LENGTH = 3
SEQUENCE_OVERLAP = 1
DATA_DIRECTORY = "./all_respeck"
MODEL_NAME = "CNN_2.keras"


#LSTM MODEL
def train_model_LSTM(sequences, labels_encoded, unique_labels, epochs, batch_size):
# Define the LSTM model
    model = Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(75, 6)), # input shape is (sequence length * 25, 6) 
        layers.LSTM(64),
        layers.Dense(len(unique_labels), activation='softmax') 
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #  training model
    model.fit(sequences, labels_encoded, epochs, batch_size)

    return model 

#CNN MODEL
def train_model_CNN(input_data, labels_encoded, unique_labels, epochs, batch_size, validation_data):
    # Define the CNN model for your specific input shape
    model = Sequential([
        layers.Conv1D(32, 3, activation='relu', input_shape=(75, 6)),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the CNN model
    model.fit(input_data, labels_encoded, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    return model


# split data into training, dev, and test sets
def train_dev_test_split(data, labels, dev_size, test_size, random_state=30):
    # Split the data into training and temporary (dev + test) sets
    train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=(dev_size + test_size), random_state=random_state)
    
    # Split the temporary data into dev and test sets
    dev_data, test_data, dev_labels, test_labels = train_test_split(temp_data, temp_labels, 
                                                                 test_size=(test_size / (dev_size + test_size)), random_state=random_state)
    
    return train_data, dev_data, test_data, train_labels, dev_labels, test_labels



UNIQUE_LABELS = ['misc_movements&normal_breathing', 'sitting&singing', 'standing&talking', 'sitting&normal_breathing', 'standing&laughing', 'lying_down_back&talking', 'standing&normal_breathing', 'lying_down_back&coughing', 'standing&singing', 'shuffle_walking&normal_breathing', 'descending_stairs&normal_breathing', 'sitting&eating', 'standing&coughing', 'lying_down_stomach&normal_breathing', 'lying_down_stomach&talking', 'lying_down_left&hyperventilating', 'sitting&hyperventilating', 'lying_down_back&singing', 'lying_down_right&hyperventilating', 'walking&normal_breathing', 'sitting&coughing', 'sitting&talking', 'lying_down_right&coughing', 'lying_down_stomach&hyperventilating', 'lying_down_left&normal_breathing', 'standing&hyperventilating', 'lying_down_stomach&laughing', 'lying_down_left&coughing', 'standing&eating', 'running&normal_breathing', 'lying_down_stomach&singing', 'lying_down_back&hyperventilating', 'lying_down_back&normal_breathing', 'lying_down_right&normal_breathing', 'lying_down_left&laughing', 'lying_down_left&talking', 'ascending_stairs&normal_breathing', 'lying_down_right&laughing', 'lying_down_right&singing', 'lying_down_right&talking', 'lying_down_back&laughing', 'sitting&laughing', 'lying_down_stomach&coughing', 'lying_down_left&singing']
if __name__=="__main__":

    tagged_sequences = training_data_generator.generate_training_data(DATA_DIRECTORY, SEQUENCE_LENGTH, SEQUENCE_OVERLAP)

    # Combine all sequences and labels
    sequences = [sequence for _, sequence in tagged_sequences]
    labels = [label for label, _ in tagged_sequences]


    # encode labels to numbers
    sequences = np.array(sequences, dtype=np.float32)
    label_to_index = {label: idx for idx, label in enumerate(UNIQUE_LABELS)}
    labels_encoded = [label_to_index[label] for label in labels]
    labels_encoded = np.array(labels_encoded)
    print(f'---------{len(labels_encoded)}')

    train_data, dev_data, test_data, train_labels, dev_labels, test_labels = train_dev_test_split(sequences, labels_encoded, dev_size=0.1, test_size=0.1) #10% dev, 10% test



    # train and save model (CHOOSE BETWEEN CNN AND LSTM)
    model = train_model_CNN(train_data, train_labels, UNIQUE_LABELS, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(dev_data, dev_labels)) #batch_size, epochs

    # Save the trained model
    model.save("models/" + MODEL_NAME)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


    

