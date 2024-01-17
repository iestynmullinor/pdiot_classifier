import argparse
import tensorflow as tf
from keras import models
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

# All models can classify all classes which are relevant to their specific task
# In the preprocessing stage, we remove all classes from the test data which are not relevant to the current task




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

TASK2_CLASSES = ["sitStand_breathingNormal",
    "sitStand_coughing",
    "sitStand_hyperventilating",

    "lyingLeft_breathingNormal",
    "lyingLeft_coughing",
    "lyingLeft_hyperventilating",

    "lyingRight_breathingNormal",
    "lyingRight_coughing",
    "lyingRight_hyperventilating",

    "lyingBack_breathingNormal",
    "lyingBack_coughing",
    "lyingBack_hyperventilating",

    "lyingStomach_breathingNormal",
    "lyingStomach_coughing",
    "lyingStomach_hyperventilating",]

TASK3_CLASSES = ["sitStand_breathingNormal",
    "sitStand_coughing",
    "sitStand_hyperventilating",
    "sitStand_other",
    
    "lyingLeft_breathingNormal",
    "lyingLeft_coughing",
    "lyingLeft_hyperventilating",
    "lyingLeft_other",
    
    "lyingRight_breathingNormal",
    "lyingRight_coughing",
    "lyingRight_hyperventilating",
    "lyingRight_other",
    
    "lyingBack_breathingNormal",
    "lyingBack_coughing",
    "lyingBack_hyperventilating",
    "lyingBack_other",
    
    "lyingStomach_breathingNormal",
    "lyingStomach_coughing",
    "lyingStomach_hyperventilating",
    "lyingStomach_other"]

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


def generate_test_data_from_recording(dataframe, window_length, overlap, normalise):
    # Get the data from the dataframe
    label = dataframe['class'].values[0]
    all_frames = dataframe[['accel_x', 'accel_y', 'accel_z']].values.tolist()

    sequences = generate_sequences(all_frames, window_length, overlap, normalise)

    return zip(sequences, [label]*len(sequences))

def generate_all_test_data(dataframes, window_length=5, overlap=0, normalise=False):
    test_data = []

    for dataframe in dataframes:
        test_data.extend(generate_test_data_from_recording(dataframe, window_length, overlap, normalise))

    return test_data

################## FEATURE EXTRACTION FUNCTIONS ##################


def fft(data):

    # Extract x, y, and z data
    x_data = data[:, 0]
    y_data = data[:, 1]
    z_data = data[:, 2]

    # Apply FFT to each axis
    x_fft = np.fft.fft(x_data)
    y_fft = np.fft.fft(y_data)
    z_fft = np.fft.fft(z_data)

    # The result is complex numbers, so you may want to take the magnitude
    x_magnitude = np.abs(x_fft)
    y_magnitude = np.abs(y_fft)
    z_magnitude = np.abs(z_fft)

    representation = []
    for i in range(len(x_magnitude)):
        representation.append([x_magnitude[i], y_magnitude[i], z_magnitude[i]]) #, x_frequencies[i], y_frequencies[i], z_frequencies[i]])

    return representation

def extract_fft(test_data):
        
        test_features = [fft(sequence) for sequence in test_data]

        return test_features

def differential(data):
        # Extract x, y, and z data
        x_data = data[:, 0]
        y_data = data[:, 1]
        z_data = data[:, 2]

        # Compute the differences between consecutive data points
        x_diff = np.diff(x_data)
        y_diff = np.diff(y_data)
        z_diff = np.diff(z_data)

        # Add a 0 at the start of the differential variables
        x_diff = np.insert(x_diff, 0, 0)
        y_diff = np.insert(y_diff, 0, 0)
        z_diff = np.insert(z_diff, 0, 0)
        
        # Combine the differential values into a representation
        representation = []
        for i in range(len(x_diff)):
            representation.append([x_diff[i], y_diff[i], z_diff[i]])

        return representation

def extract_differentials(test_data):
        
        test_features = [differential(sequence) for sequence in test_data]

        return test_features

def derivative(data):
        # Extract x, y, and z data
        x_data = data[:, 0]
        y_data = data[:, 1]
        z_data = data[:, 2]

        # Compute the derivative of the data
        x_derivative = np.gradient(x_data)
        y_derivative = np.gradient(y_data)
        z_derivative = np.gradient(z_data)

        # Combine the derivative values into a representation
        representation = []
        for i in range(len(x_derivative)):
            representation.append([x_derivative[i], y_derivative[i], z_derivative[i]])

        return representation

def extract_gradients(test_data):
    test_features = [derivative(sequence) for sequence in test_data]

    return test_features

def extract_features(test_data):
    fft_features = extract_fft(test_data)
    differential_features = extract_differentials(test_data)
    gradient_features = extract_gradients(test_data)

    test_features = [np.concatenate((test_data[i], fft_features[i], differential_features[i], gradient_features[i]), axis=1) for i in range(len(fft_features))]

    return test_features


################## EVALUATION FUNCTIONS ##################

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

    print("Classification report for task 1:")

    print(classification_report(test_labels, predicted_labels))


def prep_data_for_task_2(labelled_sequences):
    task2_labelled_sequences = [labelled_sequences[i] for i in range(len(labelled_sequences)) if labelled_sequences[i][1] in TASK2_CLASSES]
    task_2_test_data = [sequence for sequence, label in task2_labelled_sequences]
    task_2_test_labels = [label for sequence, label in task2_labelled_sequences]

    task_2_features = extract_features(task_2_test_data)

    return task_2_features, task_2_test_labels

def eval_task_2(model, labelled_sequences):

    test_data, test_labels = prep_data_for_task_2(labelled_sequences)

    # convert to numpy array
    test_data = np.array(test_data)

    predictions = model.predict(test_data)

    predicted_labels = np.argmax(predictions, axis=1)
    predicted_labels = [TASK2_CLASSES[i] for i in predicted_labels]

    print("Classification report for task 2:")

    print(classification_report(test_labels, predicted_labels))

def prep_data_for_task_3(labelled_sequences):
    task3_labelled_sequences = [labelled_sequences[i] for i in range(len(labelled_sequences)) if labelled_sequences[i][1] in TASK3_CLASSES]
    task_3_test_data = [sequence for sequence, label in task3_labelled_sequences]
    task_3_test_labels = [label for sequence, label in task3_labelled_sequences]

    task_3_features = extract_features(task_3_test_data)

    return task_3_features, task_3_test_labels

def eval_task_3(model, labelled_sequences):
         
        test_data, test_labels = prep_data_for_task_3(labelled_sequences)
    
        # convert to numpy array
        test_data = np.array(test_data)
    
        predictions = model.predict(test_data)
    
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_labels = [TASK3_CLASSES[i] for i in predicted_labels]

        print("Classification report for task 3:")
    
        print(classification_report(test_labels, predicted_labels))






data_recordings = load_test_data(test_data_path)


# EVALUATION FOR TASK 1

# LOADING DATA RECORDINGS (UNNORMALISED)
raw_labelled_sequences = generate_all_test_data(data_recordings)


# LOADING TASK 1 MODEL
task1_model = models.load_model(f"{model_path}/task1_model.keras")


# EVALUATING TASK 1 MODEL
print("Evaluating task 1 model...")
eval_task_1(task1_model, raw_labelled_sequences)

# FOR TASK 2 AND 3, WE NEED TO NORMALISE THE DATA

# EVALUATION FOR TASK 2

# LOADING DATA RECORDINGS (NORMALISED)
normalised_labelled_sequences = generate_all_test_data(data_recordings, normalise=True)



# LOADING TASK 2 MODEL
task2_model = models.load_model(f"{model_path}/task2_model.keras")
print("Evaluating task 2 model...")
eval_task_2(task2_model, normalised_labelled_sequences)

# EVALUATION FOR TASK 3

# LOADING TASK 3 MODEL
task3_model = models.load_model(f"{model_path}/task3_model_NOTFINAL.keras")
print("Evaluating task 3 model...")
eval_task_3(task3_model, normalised_labelled_sequences)





