# -*- coding: utf-8 -*-

# import built-in module
import pathlib
import os
import csv

# import third-party modules
import optuna

# import your own module
from ..abstractmodel import AbstractModel


def get_hardware_metrics(model: AbstractModel, trial: optuna.trial.Trial):
    model.parametrize_from_trial(trial)
    hm = model.get_hardware_metrics()
    hm_dict = hm.as_dict()

    study_dir = pathlib.Path(trial.study.user_attrs["study_dir"])
    csv_file_path = study_dir / f"trial_{trial.number}" / "hardware_metrics.csv"

    if os.path.isfile(csv_file_path):
        with open(csv_file_path, "a") as f:
            csv_writer = csv.DictWriter(f, hm_dict.keys())
            csv_writer.writerows([hm_dict])
    else:
        with open(csv_file_path, "w") as f:
            csv_writer = csv.DictWriter(f, hm_dict.keys())
            csv_writer.writeheader()
            csv_writer.writerows([hm_dict])
