
import numpy as np
from flask import Flask, jsonify, request
from keras.models import load_model

# RUN THIS TO HOST FLASK APP

UNIQUE_LABELS = ['misc_movements&normal_breathing', 'sitting&singing', 'standing&talking', 'sitting&normal_breathing', 'standing&laughing', 'lying_down_back&talking', 'standing&normal_breathing', 'lying_down_back&coughing', 'standing&singing', 'shuffle_walking&normal_breathing', 'descending_stairs&normal_breathing', 'sitting&eating', 'standing&coughing', 'lying_down_stomach&normal_breathing', 'lying_down_stomach&talking', 'lying_down_left&hyperventilating', 'sitting&hyperventilating', 'lying_down_back&singing', 'lying_down_right&hyperventilating', 'walking&normal_breathing', 'sitting&coughing', 'sitting&talking', 'lying_down_right&coughing', 'lying_down_stomach&hyperventilating', 'lying_down_left&normal_breathing', 'standing&hyperventilating', 'lying_down_stomach&laughing', 'lying_down_left&coughing', 'standing&eating', 'running&normal_breathing', 'lying_down_stomach&singing', 'lying_down_back&hyperventilating', 'lying_down_back&normal_breathing', 'lying_down_right&normal_breathing', 'lying_down_left&laughing', 'lying_down_left&talking', 'ascending_stairs&normal_breathing', 'lying_down_right&laughing', 'lying_down_right&singing', 'lying_down_right&talking', 'lying_down_back&laughing', 'sitting&laughing', 'lying_down_stomach&coughing', 'lying_down_left&singing']


model = load_model("test_model.keras")
app = Flask(__name__)

def prepare_sequence(sequence):
    individual_sequence_array = np.array(sequence, dtype=np.float32)
    individual_sequence_array = np.expand_dims(individual_sequence_array, axis=0)
    return individual_sequence_array

def predict_result(sequence):
    predictions = model.predict(sequence)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_tag = UNIQUE_LABELS[predicted_class_index[0]]  
    return(predicted_tag)


@app.route('/predict', methods=['POST'])
def classify_activity():
    try:
    
        data = request.get_json(force=True)

        # converts list of lists to format that can be used by classifier
        sequence = prepare_sequence(data)

        # Return on a JSON format
        return jsonify(prediction=predict_result(sequence))

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/', methods=['GET'])
def index():
    return jsonify('PDIOT prediction model API')

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0')