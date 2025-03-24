import tensorflow as tf
import numpy as np

def expclip(x, a=None, b=None):
    '''
    clipping exp function to limit Sigmoid.
    Exponential soft clipping, with parameterized corner sharpness.
    '''
    # default scaling constants to match tanh corner shape
    _c_tanh = 2 / (np.e * np.e + 1)  # == 1 - np.tanh(1) ~= 0.24
    _c_softclip = np.log(2) / _c_tanh
    _c_expclip = 1 / (2 * _c_tanh)

    c = _c_expclip
    if a is not None and b is not None:
        c /= (b - a) / 2

    v = tf.clip_by_value(x, a, b)

    if a is not None:
        v = v + tf.exp(-c * np.abs(x - a)) / (2 * c)
    if b is not None:
        v = v - tf.exp(-c * np.abs(x - b)) / (2 * c)

    return v