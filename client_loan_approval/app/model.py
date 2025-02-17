import os
import joblib


def load_model(model_path):
    # Absolut path to the model
    absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), model_path))
    print('Loading model from:', absolute_path)

    # Load the model
    with open(absolute_path, 'rb') as file:
        model = joblib.load(file)

    return model
