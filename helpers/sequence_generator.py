# THIS ONE IS SEQUENCE GENERATOR

# If we take the approach of splitting each recording into shorter overlapping sequences, to make real time classification easier,
# then this takes a recording as a CSV file and splits it into sequences of a specific length with specific overlap between sequences

import csv
import numpy as np

def min_max_scaling_symmetric(data, new_max=1):
    """
    Applies Min-Max scaling to each column of the input data symmetrically around 0.
    """
    max_abs = np.max(np.abs(data), axis=0, keepdims=True)
    
    scaled_data = data / max_abs * new_max

    return scaled_data

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

def generate_sequences_from_file_without_gyroscope(filepath, length, overlap, normalise):
    return generate_sequences(open_csv_without_gyro(filepath), length, overlap, normalise)

# returns all generated sequences from a filepath
def generate_sequences_from_file_with_gyroscope(filepath, length, overlap, normalise):
    return generate_sequences(open_csv_with_gyro(filepath), length, overlap, normalise)

# if __name__ == '__main__':
#     print(generate_sequences_from_file_without_gyroscope('./all_respeck/S1_respeck_ascending stairs_normal_clean.csv', 5, 0))

