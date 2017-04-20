# Recognizing emotions from frontal-faced images

import dlib, cv2
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib

EMOTION_MAP = np.array(map(str, range(0, 9)))

def predict(batch, model_path):
    """
    Function to predict the emotion given the path to model file ( sklearn's model file )

    :param array batch: Input to the model
    :param str model_path: Path to the model file

    :return array: Labels predicted by the model
    """
    model = joblib.load(model_path)
    #if model is None:
        # Throw error
    return model.predict(batch)
