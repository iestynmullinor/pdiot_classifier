import numpy as np
import tensorflow as tf
from keras import layers, Sequential
from keras.models import load_model
from sklearn.model_selection import train_test_split

# TAKES IN 3 MODELS AND COMBINES THEM INTO ONE MODEL
# FIRST RUNS BINARY CLASSIFIER TO DETERMINE WHETHER THE ACTIVITY IS STATIONARY OR NOT
# THEN RUNS THE APPROPRIATE MODEL BASED ON THE OUTPUT OF THE BINARY CLASSIFICATION
# I HAVE NO IDEA IF THIS WORKS OR NOT


# names of models
MODELS_DIRECTORY = "./models/models_for_presentation"

# model for telling if stationary or moving
STATIONARY_MOVING_CLASSIFIER_NAME = "stationary_or_moving.keras" 

# model for telling what type of moving activity
MOVING_ACTIVITY_CLASSIFIER_NAME = "moving_classifier.keras"

# model for telling which stationary position
STATIONARY_POSITION_CLASSIFIER_NAME = "stationary_position_classifier.keras"

# We now have a model for each individual stationary position, that is used to determine the respiratory symptom or other activity
SITTING_STANDING_MODEL_NAME = "sitting_standing_model.keras"
LYING_DOWN_BACK_MODEL_NAME = "lying_down_back_model.keras"
LYING_DOWN_STOMACH_MODEL_NAME = "lying_down_stomach_model.keras"
LYING_DOWN_RIGHT_MODEL_NAME = "lying_down_right_model.keras"
LYING_DOWN_LEFT_MODEL_NAME = "lying_down_left_model.keras"




# loading models
stationary_moving_classifier = load_model(MODELS_DIRECTORY + "/" + STATIONARY_MOVING_CLASSIFIER_NAME)
moving_activity_classifier = load_model(MODELS_DIRECTORY + "/" + MOVING_ACTIVITY_CLASSIFIER_NAME)
stationary_position_classifier = load_model(MODELS_DIRECTORY + "/" + STATIONARY_POSITION_CLASSIFIER_NAME)
sitting_standing_classifier = load_model(MODELS_DIRECTORY + "/" + SITTING_STANDING_MODEL_NAME)
lying_down_back_classifier = load_model(MODELS_DIRECTORY + "/" + LYING_DOWN_BACK_MODEL_NAME)
lying_down_stomach_classifier = load_model(MODELS_DIRECTORY + "/" + LYING_DOWN_STOMACH_MODEL_NAME)
lying_down_right_classifier = load_model(MODELS_DIRECTORY + "/" + LYING_DOWN_RIGHT_MODEL_NAME)
lying_down_left_classifier = load_model(MODELS_DIRECTORY + "/" + LYING_DOWN_LEFT_MODEL_NAME)

# arrays of what the output values correspond to, this is used for finding what the output of the classifier means, 
# if output of stationary_position classifier is 0, then result is OUTCOMES_FOR_STATIONARY_POSITION_CLASSIFIER[0] = "sitting_or_standing"


OUTCOMES_FOR_STATIONARY_MOVING_CLASSIFIER = ["stationary", "moving"]

OUTCOMES_FOR_MOVING_ACTIVITY_CLASSIFIER = [
    "walking",
    "ascending_stairs",
    "descending_stairs",
    "shuffle_walking",
    "running",
    "misc_movements"]

OUTCOMES_FOR_STATIONARY_POSITION_CLASSIFIER = [
    "sitting_or_standing",
    "lying_down_back",
    "lying_down_stomach",
    "lying_down_right",
    "lying_down_left"]

OUTCOMES_SITTING_STANDING_CLASSIFIER = ["sitting_or_standing&coughing",
    "sitting_or_standing&hyperventilating",
    "sitting_or_standing&normal_breathing",
    
    
    "sitting_or_standing&talking",
    "sitting_or_standing&eating",
    "sitting_or_standing&singing",
    "sitting_or_standing&laughing"]

OUTCOMES_FOR_LYING_DOWN_CLASSIFIERS = ["lying_down_stomach&coughing",
    "lying_down_stomach&hyperventilating",
    "lying_down_stomach&talking",
    "lying_down_stomach&singing",
    "lying_down_stomach&laughing",
    "lying_down_stomach&normal_breathing"]





# idk
# Create a wrapper model
#wrapper_input = tf.keras.layers.Input(shape=(75, 3))
#output_model1 = binary_classifer(wrapper_input)

# Use a Lambda layer to select the appropriate model based on the binary output of model1
#output = tf.keras.layers.Lambda(
#    lambda x: stationary_activity_classifier(wrapper_input) if x == 1 else physical_activity_classifier(wrapper_input)
#)(output_model1)

# Define the combined model
#combined_model = tf.keras.Model(inputs=wrapper_input, outputs=output)

# Compile the combined model
#combined_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

