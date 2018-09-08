# -*- coding: utf-8 -*-

class Framework_Config(object):
    def __init__(self, framework_name: str):
        self.frame_work = framework_name

    def set_detail(self):
        raise ValueError("This function is needed to rewrite.")

class Keras_Config(Framework_Config):
    def __init__(self):
        super(Keras_Config, self).__init__("keras")

    def set_detail(self, fraction: float=None, is_auto_increase: bool=True):
        import keras.backend.tensorflow_backend as KTF
        import tensorflow as tf
        config = tf.ConfigProto()
        if fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = fraction
        config.gpu_options.allow_growth = is_auto_increase
        session = tf.Session(config=config)
        KTF.set_session(session)