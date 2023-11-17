# If we take the approach of splitting each recording into shorter overlapping sequences, to make real time classification easier,
# then this takes a recording as a CSV file and splits it into sequences of a specific length with specific overlap between sequences

import csv
import numpy as np

# Converts csv file to a list of lists, where each list is [accel_x, accel_y, accel_z] for each time stamp
# --- ONLY ACCELEROMETER DATA ---
def open_csv_without_gyro(file_path):
    data_points = []

    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        # Skip the header row 
        next(csv_reader, None)
        for row in csv_reader:
            data_points.append(row[2:5]) # Only includes accelerometer data
            
    return data_points


# Converts csv file to a list of lists, where each list is [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] for each time stamp
def open_csv_with_gyro(file_path):
    data_points = []

    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        # Skip the header row 
        next(csv_reader, None)
        for row in csv_reader:
            data_points.append(row[-6:]) # Includes accelerometer and gyroscope data
            
    return data_points


def generate_sequences(all_frames, length, overlap, normalise=False):

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
        sequence_array.append(sequence)
        sequence_start_frame = sequence_start_frame + frames_per_sequence - frames_per_overlap
        sequence_end_frame = sequence_end_frame + frames_per_sequence - frames_per_overlap

    # Normalize every value in the matrix by accounting for every other value in the sequence array
    if normalise:
        sequence_array = np.array(sequence_array, dtype=float) # convert to float
        for i in range(sequence_array.shape[0]):
            sequence_array[i] = (sequence_array[i] - np.mean(sequence_array[i], axis=0)) / np.std(sequence_array[i], axis=0)
        sequence_array = sequence_array.tolist() # convert back to list

    return np.array(sequence_array)

def generate_sequences_from_file_without_gyroscope(filepath, length, overlap, normalise=False):
    return generate_sequences(open_csv_without_gyro(filepath), length, overlap, normalise=False)

# returns all generated sequences from a filepath
def generate_sequences_from_file_with_gyroscope(filepath, length, overlap, normalise):
    return generate_sequences(open_csv_with_gyro(filepath), length, overlap, normalise)

# if __name__ == '__main__':
#     print(generate_sequences_from_file_without_gyroscope('./all_respeck/S1_respeck_ascending stairs_normal_clean.csv', 5, 0))

