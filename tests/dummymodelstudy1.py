# -*- coding: utf-8 -*-

# import built-in module
from typing import Dict, Any
import pathlib

# import third-party modules
import optuna

# import your own module
from brainmepnas import AbstractModelStudy, HardwareMetrics, AccuracyMetrics


class DummyModelStudy1(AbstractModelStudy):
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
    OBJ_1_METRIC = "sensitivity"
    OBJ_1_SCALING = lambda x: x
    OBJ_1_DIRECTION = "maximize"
    OBJ_2_METRIC = "energy"
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
        pass

    @classmethod
    def get_accuracy_metrics(cls, trial: optuna.Trial,
                             inner_fold: int) -> AccuracyMetrics:
        pass

    @classmethod
    def get_hardware_metrics(cls, trial: optuna.Trial,
                             inner_fold: int) -> HardwareMetrics:
        pass


if __name__ == "__main__":
    DummyModelStudy1.cli_entry_point()
