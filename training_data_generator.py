# This tags every file in the directory with the correct activity,
# it then generates sequences from each file, and gives each sequence the correct tag

import file_tagger
import sequence_genrator

DATA_DIRECTORY = "./all_respeck"

def generate_training_data(directory, sequence_length, overlap):

    tagged_data = []

    # group each csv file into their respective areas
    csv_dictionary = file_tagger.tag_directory(directory)

    # iterates through each activity
    for key in csv_dictionary:

        # iterates through each csv file for the activity 
        for csv_file in csv_dictionary[key]:
            sequences = sequence_genrator.generate_sequences_from_file(directory + "/" + csv_file, sequence_length, overlap)

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