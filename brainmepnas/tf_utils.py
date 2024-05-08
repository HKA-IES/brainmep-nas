# -*- coding: utf-8 -*-

# import built-in module
from typing import Literal

# import third-party modules
import tensorflow as tf
import numpy as np

# import your own module


def generate_tflite_model(keras_model: tf.keras.Model,
                          representative_input: np.ndarray,
                          input_format: Literal["float", "int8"],
                          output_format: Literal["float", "int8"]):
    """
    Convert the given keras model to a tflite model with a randomly generated
    representative dataset.

    :param keras_model: Keras model to convert.
    :param input_format: One of float or int8.
    :param output_format: One of float or int8.
    :return tflite_model
    """
    if input_format == "float":
        input_tensor_type = tf.float32
        representative_dataset_min = -1
        representative_dataset_max = 1
    elif input_format == "int8":
        input_tensor_type = tf.int8
        representative_dataset_min = -128
        representative_dataset_max = 127
    else:
        raise ValueError(f"input_format={input_format} is not supported.")

    if output_format == "float":
        output_tensor_type = tf.float32
    elif output_format == "int8":
        output_tensor_type = tf.int8
    else:
        raise ValueError(f"output_format={output_format} is not supported.")

    if representative_input is None:
        def representative_dataset():
            nb_samples = 100
            min_value = representative_dataset_min
            max_value = representative_dataset_max
            input_shape = keras_model.layers[0].input_shape[1:]
            for i in range(nb_samples):
                data = (max_value - min_value) * np.random.random_sample(input_shape) + min_value
                data = data.astype(np.float32)
                yield [np.expand_dims(data, axis=0)]
    else:
        def representative_dataset():
            input_shape = keras_model.layers[0].input_shape[1:]
            for x in representative_input:
                yield [np.expand_dims(x, axis=[0, 3]).astype(
                np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Should always be set to [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # Should always be set to [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = input_tensor_type  # Either tf.float32, tf.int8 (recommended), tf.uint8
    converter.inference_output_type = output_tensor_type  # Either tf.float32, tf.int8 (recommended), tf.uint8
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    return tflite_model
