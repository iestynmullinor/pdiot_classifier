import tensorflow as tf
from keras.models import load_model

MODEL_TO_CONVERT = "CNN_2.keras"
NAME_OF_TARGET = "CNN_2.tflite"


model = load_model("models/" + MODEL_TO_CONVERT)
tf.saved_model.save(model, "saved_model_keras_dir")

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_keras_dir") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open("models/" + NAME_OF_TARGET, 'wb') as f:
  f.write(tflite_model)
