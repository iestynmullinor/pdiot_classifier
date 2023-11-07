import file_tagger
import sequence_genrator
import csv
import numpy as np

path = "./forest_data/ForestData.csv"


# Tag all files in directory, returning a hashmap
# Encode the different file types with a specific integer
# For each file type, group the data in windows of 4 with an overlap of 3
# Write this data to the new csv file in forest_data and tag it with the encoded tag

# encoded_labels =

RECORDING_TYPES = [
    "lying_down_left&coughing",
    "lying_down_right&singing",
    "ascending_stairs&normal_breathing",
    "lying_down_left&laughing",
    "lying_down_back&hyperventilating",
    "shuffle_walking&normal_breathing",
    "standing&singing",
    "sitting&laughing",
    "lying_down_left&singing",
    "walking&normal_breathing",
    "sitting&hyperventilating",
    "lying_down_left&hyperventilating",
    "lying_down_left&talking",
    "lying_down_stomach&normal_breathing",
    "lying_down_back&coughing",
    "lying_down_stomach&hyperventilating",
    "lying_down_right&laughing",
    "standing&laughing",
    "lying_down_stomach&laughing",
    "descending_stairs&normal_breathing",
    "lying_down_stomach&talking",
    "lying_down_back&normal_breathing",
    "lying_down_back&laughing",
    "standing&talking",
    "standing&eating",
    "lying_down_stomach&coughing",
    "lying_down_back&talking",
    "sitting&talking",
    "lying_down_right&normal_breathing",
    "lying_down_stomach&singing",
    "lying_down_back&singing",
    "misc_movements&normal_breathing",
    "standing&normal_breathing",
    "standing&coughing",
    "lying_down_right&talking",
    "lying_down_right&coughing",
    "standing&hyperventilating",
    "sitting&normal_breathing",
    "lying_down_left&normal_breathing",
    "lying_down_right&hyperventilating",
    "sitting&coughing",
    "sitting&singing",
    "running&normal_breathing",
    "sitting&eating",
]
RECORDINGS_ENCODED = {
    recording_type: i for i, recording_type in enumerate(RECORDING_TYPES)
}


# Takes a 2D array of length 25*window_length which holds the all of the recordings for that window
# Converts the individual recordings into a string and appends it to the csv file
# Writes all recordings in a window to the same line
def append_data_to_csv_file(data, recordingType, filename=path):
    data_line = [RECORDINGS_ENCODED[recordingType]]
    
    for recording in data:
        recording_str = np.array2string(recording, separator=',', max_line_width=np.inf)
        data_line.append(recording_str)

    with open(filename, mode="a") as file:
        writer = csv.writer(file)
        writer.writerow(data_line)
        


def generate_forest_data(gyro, window_length, overlap, path="./all_respeck"):
    # Open the file in write mode to wipe its contents
    with open(path + "/ForestData.csv", mode="w") as file:
        pass

    tagged_files = file_tagger.tag_directory(path)
    for key in tagged_files:
        for recording in tagged_files[key]:
            filepath = path + "/" + recording
            if gyro:
                sequences = (sequence_genrator.generate_sequences_from_file_with_gyroscope(filepath, window_length, overlap))
            else:
                sequences = (sequence_genrator.generate_sequences_from_file_without_gyroscope(filepath, window_length, overlap))
                
            for sequence in sequences:
                append_data_to_csv_file(sequence, key)


if __name__ == "__main__":
    generate_forest_data(False, 4, 3)
