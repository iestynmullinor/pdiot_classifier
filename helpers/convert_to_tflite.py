import tensorflow as tf
from keras.models import load_model

NAME_OF_MODEL_TO_CONVERT = "task3_model_final"

ORIGINAL_MODEL = NAME_OF_MODEL_TO_CONVERT + ".keras"
NAME_OF_TARGET = NAME_OF_MODEL_TO_CONVERT + ".tflite"


model = load_model("models/presentation_models/" + ORIGINAL_MODEL)
tf.saved_model.save(model, "models/presentation_models/")

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_keras_dir") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open("models/presentation_models/" + NAME_OF_TARGET, 'wb') as f:
  f.write(tflite_model)
