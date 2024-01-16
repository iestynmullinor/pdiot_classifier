import argparse
import tensorflow as tf
from keras import models
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report



TASK1_CLASSES = ["sitStand_breathingNormal",
    "lyingBack_breathingNormal",
    "lyingStomach_breathingNormal",
    "lyingRight_breathingNormal",
    "lyingLeft_breathingNormal",
    "normalWalking_breathingNormal",
    "ascending_breathingNormal",
    "descending_breathingNormal",
    "shuffleWalking_breathingNormal",
    "running_breathingNormal",
    "miscMovement_breathingNormal"]

TASK2_CLASSES = ["sitting_or_standing&normal_breathing",
    "sitting_or_standing&coughing",
    "sitting_or_standing&hyperventilating",

    "lying_down_left&normal_breathing",
    "lying_down_left&coughing",
    "lying_down_left&hyperventilating",

    "lying_down_right&normal_breathing",
    "lying_down_right&coughing",
    "lying_down_right&hyperventilating",

    "lying_down_back&normal_breathing",
    "lying_down_back&coughing",
    "lying_down_back&hyperventilating",

    "lying_down_stomach&normal_breathing",
    "lying_down_stomach&coughing",
    "lying_down_stomach&hyperventilating",]

TASK3_CLASSES = []

# Create the parser
parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data')

# Parse the arguments
args = parser.parse_args()

model_path = args.model_path
test_data_path = args.test_data_path

def load_test_data(test_data_path):
    # Read the CSV file
    data = pd.read_csv(test_data_path)

    # Split the data into different dataframes based on the 'fileid' field
    dataframes = [group for _, group in data.groupby('file_id')]
    
    return dataframes

def generate_sequences(all_frames, length=5, overlap=0, normalise=False):

    # sequence_array is a list of every generated sequence
    # a sequence is of form [frame1, frame2, ...] where each frame is of form [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    sequence_array = []

    #no of data points in a sequence or timetamp, since 25Hz
    frames_per_sequence = length*25 
    frames_per_overlap = overlap*25
    total_frames = len(all_frames)

    # initialising algorithm for generating sequences with overlaps
    sequence_start_frame = 0
    sequence_end_frame = frames_per_sequence

    # adds each sequence to array
    while sequence_end_frame <= total_frames:
        sequence = all_frames[sequence_start_frame: sequence_end_frame]
        
        # Normalize every value in the sequence matrix if normalise is True (commented code)
        # Scales each column to be on scale [-1,1] while keeping the sign of each value consistent (uncommented code)
        if normalise:
            sequence = np.array(sequence, dtype=float)

            norm = np.linalg.norm(sequence, axis=0)[np.newaxis, :]
            norm[norm == 0] = 1
            sequence = sequence / norm
            sequence = sequence.tolist()
        
        sequence_array.append(sequence)
        
        sequence_start_frame = sequence_start_frame + frames_per_sequence - frames_per_overlap
        sequence_end_frame = sequence_end_frame + frames_per_sequence - frames_per_overlap

    return np.array(sequence_array)


def generate_test_data_from_recording(dataframe):
    # Get the data from the dataframe
    label = dataframe['class'].values[0]
    all_frames = dataframe[['accel_x', 'accel_y', 'accel_z']].values.tolist()

    sequences = generate_sequences(all_frames)

    return zip(sequences, [label]*len(sequences))

def generate_all_test_data(dataframes):
    test_data = []

    for dataframe in dataframes:
        test_data.extend(generate_test_data_from_recording(dataframe))

    return test_data

def prep_data_for_task_1(labelled_sequences):

    task1_labelled_sequences = [labelled_sequences[i] for i in range(len(labelled_sequences)) if labelled_sequences[i][1] in TASK1_CLASSES]
    task_1_test_data = [sequence for sequence, label in task1_labelled_sequences]
    task_1_test_labels = [label for sequence, label in task1_labelled_sequences]

    return task_1_test_data, task_1_test_labels

def eval_task_1(model, labelled_sequences):

    test_data, test_labels = prep_data_for_task_1(labelled_sequences)

    # convert to numpy array
    test_data = np.array(test_data)

    predictions = model.predict(test_data)

    predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels = [TASK1_CLASSES[i] for i in predicted_labels]

    print(classification_report(test_labels, predicted_labels))






    



# LOADING DATA RECORDINGS
data_recordings = load_test_data(test_data_path)
labelled_sequences = generate_all_test_data(data_recordings)


# LOADING TASK 1 MODEL
task1_model = models.load_model(f"{model_path}/task1_model.keras")


# EVALUATING TASK 1 MODEL
print("Evaluating task 1 model...")
eval_task_1(task1_model, labelled_sequences)

# labelled_sequences is a list of tuples of form (sequence, label) 
# the labels have the format of the sample test set so will need to keep this in mind




# Load the model
# model = models.load_model(model_path)




