# This tags every file in the directory with the correct activity,
# it then generates sequences from each file, and gives each sequence the correct tag

import file_tagger
import helpers.sequence_generator as sequence_generator

DATA_DIRECTORY = "./all_respeck"

def generate_training_data(directory, sequence_length, overlap, gyro = True): # if gyro is false, only accelerometer data is used

    tagged_data = []

    # group each csv file into their respective areas
    csv_dictionary = file_tagger.tag_directory(directory)

    # iterates through each activity
    for key in csv_dictionary:

        # iterates through each csv file for the activity 
        for csv_file in csv_dictionary[key]:
            if gyro:
                sequences = sequence_generator.generate_sequences_from_file_with_gyroscope(directory + "/" + csv_file, sequence_length, overlap)
            else:
                sequences = sequence_generator.generate_sequences_from_file_without_gyroscope(directory + "/" + csv_file, sequence_length, overlap)

            # iterate through each generated sequence
            for sequence in sequences:
                tagged_data.append((key, sequence))

    print ("there are " + str(len(tagged_data)) + " tagged sequences in the dataset")
    return tagged_data

# for testing
if __name__ == "__main__":
    file_path = "output.txt"
    with open(file_path, "w") as file:
        file.write(str(generate_training_data()))