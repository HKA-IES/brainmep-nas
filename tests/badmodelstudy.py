# -*- coding: utf-8 -*-

# import built-in module
from typing import Dict, Any, Literal, Optional
import pathlib

# import third-party modules
import optuna

# import your own module
from brainmepnas import (AbstractModelStudy, HardwareMetrics, AccuracyMetrics,
                         CombinedMetrics)


class BadModelStudy(AbstractModelStudy):
    """
    Dummy implementation of AbstractModelStudy to facilitate testing.
    """

    @classmethod
    def _sample_search_space(cls, trial: optuna.Trial) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def _get_accuracy_metrics(cls, trial: optuna.Trial,
                              trial_dir: pathlib.Path,
                              loop: Literal["inner", "outer"],
                              outer_fold: int,
                              inner_fold: Optional[int] = None) -> AccuracyMetrics:
        raise NotImplementedError

    @classmethod
    def _get_hardware_metrics(cls, trial: optuna.Trial,
                              trial_dir: pathlib.Path,
                              loop: Literal["inner", "outer"],
                              outer_fold: int,
                              inner_fold: Optional[int] = None) -> HardwareMetrics:
        raise NotImplementedError

    @classmethod
    def _get_combined_metrics(cls, accuracy_metrics: AccuracyMetrics,
                              hardware_metrics: HardwareMetrics) -> CombinedMetrics:
        raise NotImplementedError


if __name__ == "__main__":
    BadModelStudy.cli_entry_point()
