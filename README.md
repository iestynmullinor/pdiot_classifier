# EXPLANATION FOR HOW THE MODELS COMBINE

Models for demo are found in ```models/models_for_presentation/```

### Determine if  Stationary or Moving
- Data is first put into ```stationary_moving_classifier.tflite``` to determine what model to go in next
- Output will be ```0``` or ```1```, corresponding to the values
  ``` ["stationary", "moving"] ```
- for example output of ```0``` is stationary as that is in positon 0 of the list above

### If output is "moving"
- Data is then put into ```moving_activity_classifier.tflite``` to determine which moving activity
- ouput value will be int which corresponds to one of the items in the list ```["walking", "ascending_stairs", "descending_stairs", "shuffle_walking", "running", "misc_movements"]```
- all of these are normal breathing

### If output is "stationary"
- Data is then put into ```stationary_position_classifier.tflite``` to determine which stationary position it is
- output value will be int which corresponds to one of the items in the list ```["sitting_or_standing","lying_down_back","lying_down_stomach","lying_down_right","lying_down_left"]```
- Output of this determines which classifier to enter data into, for example ```2``` would be ```lying_down_stomach_model.tflite``` since 2 corresponds to ```lying_down_stomach``` in the list above

### Output for determining respiratory activity
- For ```sitting_or_standing``` , output will be an int which corresponds to one of the items in the list ```["sitting_or_standing&coughing", "sitting_or_standing&hyperventilating", "sitting_or_standing&normal_breathing", "sitting_or_standing&talking","sitting_or_standing&eating", "sitting_or_standing&singing","sitting_or_standing&laughing"]```
- For all lying down positions, output will be int which corresponds to one of the items in the list ```["coughing","hyperventilating","talking","singing","laughing","normal_breathing"]```
  
