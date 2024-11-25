# -*- coding: utf-8 -*-

# import built-in module
import tempfile
from typing import Literal, Optional

# import third-party modules
import tensorflow as tf
import numpy as np

# import your own module


def generate_tflite_model(keras_model: tf.keras.Model,
                          input_format: Literal["float", "int8"],
                          output_format: Literal["float", "int8"],
                          representative_input: Optional[np.ndarray] = None):
    """
    Convert the given keras model to a tflite model.

    Parameters
    ----------
    keras_model: tf.keras.Model
        Keras model to convert.
    input_format: Literal["float", "int8"]
        Format of the input to the keras model.
    output_format: Literal["float", "int8"]
        Format of the output of the keras model.
    representative_input: np.ndarray, optional
        Representative input data, used to perform the quantization. If not
        given, quantization is performed using randomly generated data between
        -1 and 1 (float) or -128 and 127 (int8).

    Returns
    -------
    tflite_model: tf.keras.Model
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

    keras_model.build()

    if representative_input is None:
        def representative_dataset():
            nb_samples = 100
            min_value = representative_dataset_min
            max_value = representative_dataset_max
            input_shape = keras_model.input_shape[1:]
            for i in range(nb_samples):
                data = (max_value - min_value) * np.random.random_sample(input_shape) + min_value
                data = data.astype(np.float32)
                yield [np.expand_dims(data, axis=0)]
    else:
        def representative_dataset():
            for x in representative_input:
                yield [np.expand_dims(x, axis=[0, 3]).astype(
                    np.float32)]

    # Bug: tensorflow 2.16.1
    # converter.convert() raises AttributeError, see
    # https://github.com/tensorflow/tensorflow/issues/63867
    # Fix is to use save model.
    # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        keras_model.export(tmp_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Should always be set to [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # Should always be set to [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = input_tensor_type  # Either tf.float32, tf.int8 (recommended), tf.uint8
        converter.inference_output_type = output_tensor_type  # Either tf.float32, tf.int8 (recommended), tf.uint8
        converter.representative_dataset = representative_dataset
        tflite_model = converter.convert()

    return tflite_model
