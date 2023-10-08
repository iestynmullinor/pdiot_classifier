import numpy as np
import tensorflow as tf
import training_data_generator
import trial_LSTM_model

## RUN THIS TO TRAIN CLASSIFIER

UNIQUE_LABELS = ['misc_movements&normal_breathing', 'sitting&singing', 'standing&talking', 'sitting&normal_breathing', 'standing&laughing', 'lying_down_back&talking', 'standing&normal_breathing', 'lying_down_back&coughing', 'standing&singing', 'shuffle_walking&normal_breathing', 'descending_stairs&normal_breathing', 'sitting&eating', 'standing&coughing', 'lying_down_stomach&normal_breathing', 'lying_down_stomach&talking', 'lying_down_left&hyperventilating', 'sitting&hyperventilating', 'lying_down_back&singing', 'lying_down_right&hyperventilating', 'walking&normal_breathing', 'sitting&coughing', 'sitting&talking', 'lying_down_right&coughing', 'lying_down_stomach&hyperventilating', 'lying_down_left&normal_breathing', 'standing&hyperventilating', 'lying_down_stomach&laughing', 'lying_down_left&coughing', 'standing&eating', 'running&normal_breathing', 'lying_down_stomach&singing', 'lying_down_back&hyperventilating', 'lying_down_back&normal_breathing', 'lying_down_right&normal_breathing', 'lying_down_left&laughing', 'lying_down_left&talking', 'ascending_stairs&normal_breathing', 'lying_down_right&laughing', 'lying_down_right&singing', 'lying_down_right&talking', 'lying_down_back&laughing', 'sitting&laughing', 'lying_down_stomach&coughing', 'lying_down_left&singing']
if __name__=="__main__":

    tagged_sequences = training_data_generator.generate_training_data()

    # Combine all sequences and labels
    sequences = [sequence for _, sequence in tagged_sequences]
    labels = [label for label, _ in tagged_sequences]

    # encode labels to numbers
    sequences = np.array(sequences, dtype=np.float32)
    label_to_index = {label: idx for idx, label in enumerate(UNIQUE_LABELS)}
    labels_encoded = [label_to_index[label] for label in labels]
    labels_encoded = np.array(labels_encoded)



    # train and save model
    trial_LSTM_model.train_and_save_model(sequences, labels_encoded,UNIQUE_LABELS, 6, 30) #batch_size, epochs

    

