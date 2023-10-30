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
MODELS_DIRECTORY = "./models"
BINARY_CLASSIFIER_NAME = "binary_classifier.keras" 
PHYSICAL_ACTIVITY_CLASSIFIER_NAME = "physical_activity_without_gyro_9534.keras"
STATIONARY_ACTIVITY_CLASSIFIER_NAME = "placeholder_no_gyro.keras"


# loading models
binary_classifer = load_model(MODELS_DIRECTORY + "/" + BINARY_CLASSIFIER_NAME)
physical_activity_classifier = load_model(MODELS_DIRECTORY + "/" + PHYSICAL_ACTIVITY_CLASSIFIER_NAME)
stationary_activity_classifier = load_model(MODELS_DIRECTORY + "/" + STATIONARY_ACTIVITY_CLASSIFIER_NAME)

# idk
# Create a wrapper model
wrapper_input = tf.keras.layers.Input(shape=(75, 3))
output_model1 = binary_classifer(wrapper_input)

# Use a Lambda layer to select the appropriate model based on the binary output of model1
output = tf.keras.layers.Lambda(
    lambda x: stationary_activity_classifier(wrapper_input) if x == 1 else physical_activity_classifier(wrapper_input)
)(output_model1)

# Define the combined model
combined_model = tf.keras.Model(inputs=wrapper_input, outputs=output)

# Compile the combined model
combined_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

