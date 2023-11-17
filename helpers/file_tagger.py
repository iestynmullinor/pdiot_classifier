# This is used to identify the tag of each data recording, ie will take Respeck_s2061990_Ascending stairs_Normal_clean_28-09-2023_15-06-46.csv , and return the tag "Asending Stairs Normal Breathing"

import os
import json

# dictionary of how activities are named in the CSV files
activities_dict = {
    "_ascending_": "ascending_stairs",
    "_descending_": "descending_stairs",
    "_lyingBack_": "lying_down_back",
    "_lyingStomach_": "lying_down_stomach",
    "_lyingLeft_": "lying_down_left",
    "_lyingRight_": "lying_down_right",
    "_miscMovement_": "misc_movements",
    "_normalWalking_": "walking",
    "_running_": "running",
    "_shuffleWalking_": "shuffle_walking",
    "_sitting_": "sitting",
    "_standing_": "standing",
}
              

# dictionary of how respitory activities are names in the CSV files
resp_dict = {"_breathingNormal": "normal_breathing",
             "_laughing": "laughing",
             "_talking": "talking",
             "_singing": "singing",
             "_hyperventilating": "hyperventilating",
             "_eating": "eating",
             "_coughing": "coughing",
}

# tags one file from given name
def tag_file(filename):

    file_activity = "activity_not_found"
    file_resp = "resp_not_found"

    # find activity of file
    for activity in activities_dict.keys():
        if activity in filename:
            file_activity = activities_dict[activity]
    
    # find resp of file 
    for resp in resp_dict.keys():
        if resp in filename:
            file_resp = resp_dict[resp]
    
    return file_activity + "&" + file_resp

# iterating through files in DATA_DIRECTORY
# returns dictionary of each tag and a list of files associated with that tag
def tag_directory(data_directory):

    
    tagged_files={}
    
    for filename in os.listdir(data_directory):
        tag = tag_file(filename)

        # if tag already in dictionary
        if tag in tagged_files.keys():
            tagged_files[tag].append(filename)
        
        # if tag not yet in dictionary
        else:
            tagged_files[tag]=[filename]

    return tagged_files

# prints in a non shit way
def formatted_print(dict_of_lists):
    formatted_data = json.dumps(dict_of_lists, indent=4)
    print(formatted_data)

# only runs if this is the main program being run and not an import
if __name__ == "__main__": 
    tagged_files = tag_directory("./all_respeck")       
    formatted_print(tagged_files)
    print(tagged_files.keys())
    print(len((tagged_files.keys())))
    
