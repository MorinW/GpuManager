# -*- coding: utf-8 -*-

def set_keras(fraction: float=None, is_auto_increase: bool=True):
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    config = tf.ConfigProto()
    if fraction is not None:
        config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.gpu_options.allow_growth = is_auto_increase
    session = tf.Session(config=config)
    KTF.set_session(session)