# -*- coding: utf-8 -*-

# import built-in module
import configparser
import tempfile

# import third-party modules
import numpy as np
import optuna
import tensorflow as tf

# import your own module
from brainmepnas.modelstudy import setup_model_study
from brainmepnas import AbstractModel
from brainmepnas.hardwaremetrics_implementations import TestbenchHardwareMetrics, MltkHardwareMetrics
from brainmepnas.tf_utils import generate_tflite_model

"""
Example - NAS process for the EpiDeNet model

Sampler: Random

...
"""


class Epidenet(AbstractModel):
    """
    EpiDeNet model.
    """

    def __init__(self, optimizer=None, loss=None, metrics=None, callbacks=None):
        if optimizer is None:
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=1*10**(-4),
                                             beta_1=0.9, beta_2=0.999)
        else:
            self._optimizer = optimizer

        if loss is None:
            self._loss = tf.keras.losses.BinaryCrossentropy()
        else:
            self._loss = loss

        if metrics is None:
            self._metrics = [tf.keras.metrics.AUC(num_thresholds=200,
                                                curve='ROC',
                                                name="auc_roc")]
        else:
            self._metrics = metrics

        if callbacks is None:
            self._callbacks = []
        else:
            self._callbacks = callbacks

        self._model = None

    def parametrize(self, input_shape: tuple[int, int] = (1024, 4),
                 conv_padding: str = "same",
                 conv_1_filters: int = 4, conv_2_filters: int = 16,
                 conv_3_filters: int = 16, conv_4_filters: int = 16,
                 conv_5_filters: int = 16,
                 conv_1_kernel_size: tuple[int, int] = (4, 1),
                 conv_2_kernel_size: tuple[int, int] = (16, 1),
                 conv_3_kernel_size: tuple[int, int] = (8, 1),
                 conv_4_kernel_size: tuple[int, int] = (1, 16),
                 conv_5_kernel_size: tuple[int, int] = (1, 8),
                 pool_1_pool_size: tuple[int, int] = (8, 1),
                 pool_2_pool_size: tuple[int, int] = (4, 1),
                 pool_3_pool_size: tuple[int, int] = (4, 1),
                 pool_4_pool_size: tuple[int, int] = (1, 4)):
        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.InputLayer((input_shape[0], input_shape[1], 1)))

        # Block 1
        model.add(tf.keras.layers.Conv2D(filters=conv_1_filters,
                                         kernel_size=conv_1_kernel_size,
                                         padding=conv_padding))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_1_pool_size))

        # Block 2
        model.add(tf.keras.layers.Conv2D(filters=conv_2_filters,
                                         kernel_size=conv_2_kernel_size,
                                         padding=conv_padding))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_2_pool_size))

        # Block 3
        model.add(tf.keras.layers.Conv2D(filters=conv_3_filters,
                                         kernel_size=conv_3_kernel_size,
                                         padding=conv_padding))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_3_pool_size))

        # Block 4
        model.add(tf.keras.layers.Conv2D(filters=conv_4_filters,
                                         kernel_size=conv_4_kernel_size,
                                         padding=conv_padding))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_4_pool_size))

        # Block 5

        model.add(tf.keras.layers.Conv2D(filters=conv_5_filters,
                                         kernel_size=conv_5_kernel_size,
                                         padding=conv_padding))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
        pool_5_pool_size = (model.output.shape[1], model.output.shape[2])
        model.add(tf.keras.layers.AveragePooling2D(pool_size=pool_5_pool_size))

        # Output
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation("sigmoid"))

        model.compile(optimizer=self._optimizer,
                      loss=self._loss,
                      weighted_metrics=self._metrics)

        self._model = model

    def parametrize_from_trial(self, trial: optuna.Trial):
        """
        Determine the parameters values from an Optuna trial.

        :param trial:
        """
        conv_1_filters = trial.suggest_int("conv_1_filters", 4, 32,
                                           step=2)  # epidenet = 4
        conv_2_filters = trial.suggest_int("conv_2_filters", 4, 32,
                                           step=2)  # epidenet = 16
        conv_3_filters = trial.suggest_int("conv_3_filters", 4, 32,
                                           step=2)  # epidenet = 16
        conv_4_filters = trial.suggest_int("conv_4_filters", 4, 32,
                                           step=2)  # epidenet = 16
        conv_5_filters = trial.suggest_int("conv_5_filters", 4, 32,
                                           step=2)  # epidenet = 16
        conv_1_kernel_size = (
        trial.suggest_int("conv_1_kernel_size", 2, 32, step=2),
        1)  # baseline=(4,1)
        conv_2_kernel_size = (
        trial.suggest_int("conv_2_kernel_size", 2, 32, step=2),
        1)  # baseline=(16,1)
        conv_3_kernel_size = (
        trial.suggest_int("conv_3_kernel_size", 2, 32, step=2),
        1)  # baseline=(8,1)
        conv_4_kernel_size = (1, trial.suggest_int("conv_4_kernel_size", 2, 32,
                                                   step=2))  # baseline=(1,16)
        conv_5_kernel_size = (1, trial.suggest_int("conv_5_kernel_size", 2, 32,
                                                   step=2))  # baseline=(1,8)
        pool_1_pool_size = (
        trial.suggest_int("pool_1_pool_size", 2, 16, step=2),
        1)  # baseline=(8,1)
        pool_2_pool_size = (
        trial.suggest_int("pool_2_pool_size", 2, 16, step=2),
        1)  # baseline=(4,1)
        max_pool_size = 1024 / (pool_1_pool_size[0] * pool_2_pool_size[0])
        pool_3_pool_size = (
        trial.suggest_int("pool_3_pool_size", 2, max_pool_size, step=2),
        1)  # baseline=(4,1)
        pool_4_pool_size = (1, trial.suggest_int("pool_4_pool_size", 2, 4,
                                                 step=2))  # baseline=(1,4)

        self.parametrize((1024, 4), "same",
                         conv_1_filters, conv_2_filters,
                         conv_3_filters, conv_4_filters, conv_5_filters,
                         conv_1_kernel_size, conv_2_kernel_size,
                         conv_3_kernel_size, conv_4_kernel_size,
                         conv_5_kernel_size, pool_1_pool_size,
                         pool_2_pool_size, pool_3_pool_size,
                         pool_4_pool_size)

    def save(self, file_path):
        self._model.save(file_path)

    def load(self, file_path):
        self._model = tf.keras.models.load_model(file_path)

    def fit(self, x, y, **kwargs) -> tf.keras.callbacks.History:

        history = self._model.fit(x, y, callbacks=self._callbacks,
                                  batch_size=256, validation_split=0.2,
                                  epochs=200, shuffle=False, **kwargs)

        return history

    def predict(self, x):
        return self._model.predict(x)

    def get_hardware_metrics(self):
        config = configparser.ConfigParser()
        config.read("remotetestbench.ini")
        tflite_model = generate_tflite_model(self._model,
                                             representative_input=None,
                                             input_format="float",
                                             output_format="float")
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete_on_close=False) as f:
            f.write(tflite_model)
            f.close()
            #hm = TestbenchHardwareMetrics(host=config["credentials"]["host"],
            #                              user=config["credentials"]["user"],
            #                              password=config["credentials"]["password"],
            #                              tflite_model_path=f.name)
            hm = MltkHardwareMetrics(f.name)

        return hm

    def get_accuracy_metrics(self):


if __name__ == "__main__":
    BASE_DIR = ""
    NAME = "epidenet_nas"
    SAMPLER = optuna.samplers.RandomSampler(seed=42)
    TRAIN_DATA = [np.random.random(100) for _ in range(3)]
    TEST_DATA = [np.random.random(100) for _ in range(3)]
    NB_TRIALS = 500
    NB_GPUS = 0
    MODEL = Epidenet()

    setup_model_study(BASE_DIR, NAME, SAMPLER, TRAIN_DATA, TEST_DATA, NB_TRIALS, NB_GPUS, MODEL)
