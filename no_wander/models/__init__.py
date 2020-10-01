from tensorflow import keras

from .build import *
from .constants import LAYER_ENCODER, LAYER_POSITION_ENCODING


def load_model(filepath):
    return keras.models.load_model(filepath)
