# -*- coding: utf-8 -*-

# import built-in module
from typing import Dict, Any, Optional
import pathlib

# import third-party modules
import optuna
import numpy as np

# import your own module
from brainmepnas import (AbstractModelStudy, HardwareMetrics, AccuracyMetrics,
                            CombinedMetrics)
from brainmepnas.dataset import Dataset


class BaseModelStudy(AbstractModelStudy):
    """
    Example model study implementation. This implementation should not be used
    directly as a model study. It must be inherited from in another class which
    defines the CPU/GPU behavior.

    Note: DATASET_DIR must be set to the location of the pre-processed CHB-MIT
    dataset (see example pre_process_chb_mit).
    """
    ###########################
    # CHANGE PARAMETERS BELOW #
    ###########################

    # path to pre-processed dataset
    DATASET_DIR = pathlib.Path("/mnt/c/Users/larochelle/data/brainmepnas_dataset/examples_chbmit_time_series")

    ###########################
    # CHANGE PARAMETERS ABOVE #
    ###########################

    # To be implemented by studies inheriting from this one
    # NAME: str
    # BASE_DIR: pathlib.Path
    # THIS_FILE = __file__
    # N_PARALLEL_GPU_JOBS: int
    # N_PARALLEL_CPU_JOBS: int
    # GET_ACCURACY_METRICS_USE_GPU: bool
    # GET_HARDWARE_METRICS_USE_GPU: bool

    SAMPLER = optuna.samplers.TPESampler(seed=42)
    N_FOLDS = 5     # Patient 5 has 5 records with a seizure.
    N_TRIALS = 5    # Small to ensure example runs relatively fast.
    GET_ACCURACY_METRICS_CALL = "per_inner_fold"
    GET_HARDWARE_METRICS_CALL = "once"
    OBJ_1_METRIC = "sample_sensitivity"
    OBJ_1_SCALING = lambda x: x
    OBJ_1_DIRECTION = "maximize"
    OBJ_2_METRIC = "inference_energy"
    OBJ_2_SCALING = lambda x: (np.log10(x) + 2) / 3
    OBJ_2_DIRECTION = "minimize"

    @classmethod
    def _sample_search_space(cls, trial: optuna.Trial) -> Dict[str, Any]:
        conv_1_filters = trial.suggest_int("conv_1_filters", 4, 32, step=2)
        conv_2_filters = trial.suggest_int("conv_2_filters", 4, 32, step=2)
        conv_3_filters = trial.suggest_int("conv_3_filters", 4, 32, step=2)
        conv_4_filters = trial.suggest_int("conv_4_filters", 4, 32, step=2)
        conv_5_filters = trial.suggest_int("conv_5_filters", 4, 32, step=2)

        d = {"conv_1_filters": conv_1_filters,
             "conv_2_filters": conv_2_filters,
             "conv_3_filters": conv_3_filters,
             "conv_4_filters": conv_4_filters,
             "conv_5_filters": conv_5_filters,
             }
        return d

    @classmethod
    def get_model(cls, params: Dict[str, Any]) -> "keras.Model":
        """
        Generate the EpiDeNet architecture as a Keras Model from a dictionary
        of parameters.

        Parameters
        ----------
        params : Dict[str, Any]
            EpiDeNet architecture parameters.

        Returns
        -------
        model: keras.Model
            Keras implementation of EpiDeNet.
        """
        # Import done in the function to avoid the tf import overhead when
        # it is not needed.
        import tensorflow.keras as keras

        model = keras.Sequential()

        model.add(keras.layers.InputLayer((1024, 4, 1)))

        # Block 1
        model.add(keras.layers.Conv2D(filters=params["conv_1_filters"],
                                         kernel_size=(4, 1),
                                         padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(8, 1)))

        # Block 2
        model.add(keras.layers.Conv2D(filters=params["conv_2_filters"],
                                         kernel_size=(16, 1),
                                         padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(4, 1)))

        # Block 3
        model.add(keras.layers.Conv2D(filters=params["conv_3_filters"],
                                         kernel_size=(8, 1),
                                         padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(4, 1)))

        # Block 4
        model.add(keras.layers.Conv2D(filters=params["conv_4_filters"],
                                         kernel_size=(1, 16),
                                         padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.MaxPooling2D(pool_size=(1, 4)))

        # Block 5

        model.add(keras.layers.Conv2D(filters=params["conv_5_filters"],
                                         kernel_size=(1, 8),
                                         padding="same"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        pool_5_pool_size = (model.layers[-1].output.shape[1], model.layers[-1].output.shape[2])
        model.add(keras.layers.AveragePooling2D(pool_size=pool_5_pool_size))

        # Output
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1))
        model.add(keras.layers.Activation("sigmoid"))

        return model

    @classmethod
    def _get_accuracy_metrics(cls, trial: optuna.Trial,
                              inner_fold: Optional[int] = None) -> AccuracyMetrics:
        # Import done in the function to avoid the tf import overhead when
        # it is not needed.
        import tensorflow.keras as keras

        # ensure that the training is deterministic and always equivalent
        # (if the model would be the same)
        keras.utils.set_random_seed(42)

        # Fully deterministic results can be achieved with the following method,
        # at the cost of longer training time. Since the NAS process is already an
        # approximation, and is meant to be fast, we leave it commented.
        # tf.config.experimental.enable_op_determinism()

        # Get test and train data
        outer_fold = cls.get_outer_fold(trial.study)

        patient_5_records = [i for i in range(5)]
        patient_5_records.pop(outer_fold)

        test_seizures = [patient_5_records[inner_fold]]
        patient_5_records.pop(inner_fold)
        train_seizures = patient_5_records
        dataset = Dataset(cls.DATASET_DIR)

        params = cls._sample_search_space(trial)
        model = cls.get_model(params)# Compile
        optimizer = keras.optimizers.Adam(learning_rate=1 * 10 ** (-4),
                                             beta_1=0.9, beta_2=0.999)
        loss_fn = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.AUC(num_thresholds=200,
                                        curve='ROC',
                                        name="auc_roc")]
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      weighted_metrics=metrics)

        # Train model
        callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      patience=5,
                                                      mode="min",
                                                      start_from_epoch=10)]

        train_data = dataset.get_data({"5": train_seizures},
                                      "train",
                                      shuffle=True, shuffle_seed=42)

        model.fit(train_data[0], train_data[1],
                            callbacks=callbacks,
                            batch_size=256, validation_split=0.2,
                            epochs=5, shuffle=False)

        del train_data

        # Test model
        test_data = dataset.get_data({"5": test_seizures},
                                      "test", shuffle=False)
        test_y_predicted = model.predict(test_data[0])
        am = AccuracyMetrics(test_data[1].flatten(), test_y_predicted.flatten(),
                                           4, 2, threshold="max_f_score")

        return am

    @classmethod
    def _get_hardware_metrics(cls, trial: optuna.Trial,
                              inner_fold: Optional[int] = None) -> HardwareMetrics:
        from brainmepnas.mltkhardwaremetrics import MltkHardwareMetrics
        from brainmepnas.tf_utils import generate_tflite_model

        params = cls._sample_search_space(trial)
        model = cls.get_model(params)
        tflite_model = generate_tflite_model(model, "float", "float")

        trial_dir = cls.get_trial_dir(trial)
        tflite_model_path = trial_dir / "quantized_model.tflite"

        with open(tflite_model_path, "wb") as file:
            file.write(tflite_model)

        hm = MltkHardwareMetrics(str(tflite_model_path))

        return hm

    @classmethod
    def _get_combined_metrics(cls, accuracy_metrics: AccuracyMetrics,
                              hardware_metrics: HardwareMetrics) -> CombinedMetrics:
        cm = CombinedMetrics(accuracy_metrics, hardware_metrics)
        return cm