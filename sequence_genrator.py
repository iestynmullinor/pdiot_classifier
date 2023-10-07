# If we take the approach of splitting each recording into shorter overlapping sequences, to make real time classification easier,
# then this takes a recording as a CSV file and splits it into sequences of a specific length with specific overlap between sequences

import csv

SEQUENCE_LENGTH = 3
SEQUENCE_OVERLAP = 1

# Converts csv file to a list of lists, where each list is [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] for each time stamp
def open_csv(file_path):
    data_points = []

    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        # Skip the header row 
        next(csv_reader, None)
        for row in csv_reader:
            data_points.append(row[1:]) #currently drops the timestamp column, if we want it back then remove [1:]
            
    return data_points


def generate_sequences(all_frames, length=SEQUENCE_LENGTH, overlap=SEQUENCE_OVERLAP):

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
        sequence_array.append(all_frames[sequence_start_frame: sequence_end_frame])
        sequence_start_frame = sequence_start_frame + frames_per_sequence - frames_per_overlap
        sequence_end_frame = sequence_end_frame + frames_per_sequence - frames_per_overlap
    
    return sequence_array

# returns all generated sequences from a filepath
def generate_sequences_from_file(filepath, length=SEQUENCE_LENGTH, overlap=SEQUENCE_OVERLAP):
    return generate_sequences(open_csv(filepath), length, overlap)

