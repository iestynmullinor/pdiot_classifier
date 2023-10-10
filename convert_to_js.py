import tensorflowjs as tfjs
from keras.models import load_model

if __name__=="__main__":
    model = load_model("test_model.h5")
    tfjs.converters.save_keras_model(model, "./tensorflowjs_model")


