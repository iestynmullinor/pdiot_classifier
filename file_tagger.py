# This is used to identify the tag of each data recording, ie will take Respeck_s2061990_Ascending stairs_Normal_clean_28-09-2023_15-06-46.csv , and return the tag "Asending Stairs Normal Breathing"

import os
import json

# dictionary of how activities are named in the CSV files
activities_dict = {"_Ascending stairs_": "ascending_stairs", 
              "_Descending stairs_": "descending_stairs",
              "_Lying down back_": "lying_down_back",
              "_Lying down on left_": "lying_down_left",
              "_Lying down on stomach_": "lying_down_stomach",
              "_Lying down right_": "lying_down_right",
              "_Miscellaneous movements_": "misc_movements",
              "_Normal walking_": "walking",
              "_Running_": "running",
              "_Shuffle walking_": "shuffle_walking",
              "_Sitting_": "sitting",
              "_Standing_": "standing"}

# dictionary of how respitory activities are names in the CSV files
resp_dict = {"_Normal_": "normal_breathing",
             "_Laughing_": "laughing",
             "_Talking_": "talking",
             "_Singing_": "singing",
             "_Hyperventilating_": "hyperventilating",
             "_Eating_": "eating",
             "_Coughing_": "coughing"}

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
    formatted_print(tag_directory("./clean"))
