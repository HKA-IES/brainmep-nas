# -*- coding: utf-8 -*-

# import built-in module
from typing import Dict, Any
import pathlib

# import third-party modules
import optuna
import numpy as np

# import your own module
from brainmepnas import (AbstractModelStudy, HardwareMetrics, AccuracyMetrics,
                         CombinedMetrics)


class GoodModelStudy(AbstractModelStudy):
    """
    Dummy implementation of AbstractModelStudy to facilitate testing.
    """

    # Model study
    NAME = "dummy_model_study_1"
    SAMPLER = optuna.samplers.RandomSampler()
    BASE_DIR = pathlib.Path("dummy_model_study_1/")
    N_FOLDS = 3
    N_TRIALS = 5
    THIS_FILE = __file__

    # Objectives
    OBJ_1_METRIC = "sample_sensitivity"
    OBJ_1_SCALING = lambda x: x
    OBJ_1_DIRECTION = "maximize"
    OBJ_2_METRIC = "inference_energy"
    OBJ_2_SCALING = lambda x: 2*x
    OBJ_2_DIRECTION = "minimize"

    # Jobs queue
    N_PARALLEL_GPU_JOBS = 2
    N_PARALLEL_CPU_JOBS = 1
    GET_ACCURACY_METRICS_CALL = "per_inner_fold"
    GET_ACCURACY_METRICS_USE_GPU = True
    GET_HARDWARE_METRICS_CALL = "once"
    GET_HARDWARE_METRICS_USE_GPU = False

    @classmethod
    def _sample_search_space(cls, trial: optuna.Trial) -> Dict[str, Any]:
        d = {"param0": trial.suggest_int("param0", 0, 10),
             "param1": trial.suggest_float("param1", 2, 3)}
        return d

    @classmethod
    def _get_accuracy_metrics(cls, trial: optuna.Trial,
                              inner_fold: int) -> AccuracyMetrics:
        # Same setup as TestAccuracyMetrics.test_general_attributes()
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1, 0, 1, 0, 0])

        sample_duration = 1
        sample_offset = 1

        am = AccuracyMetrics(y_true, y_pred, sample_duration,
                             sample_offset, threshold=0.5)
        return am

    @classmethod
    def _get_hardware_metrics(cls, trial: optuna.Trial,
                              inner_fold: int) -> HardwareMetrics:
        hm = HardwareMetrics(inference_time=1,
                             inference_energy=2)
        return hm

    @classmethod
    def _get_combined_metrics(cls, accuracy_metrics: AccuracyMetrics,
                              hardware_metrics: HardwareMetrics) -> CombinedMetrics:
        cm = CombinedMetrics(accuracy_metrics, hardware_metrics)
        return cm


if __name__ == "__main__":
    GoodModelStudy.cli_entry_point()
