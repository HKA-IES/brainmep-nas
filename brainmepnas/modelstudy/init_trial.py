# -*- coding: utf-8 -*-

# import built-in module
import pathlib
import os
import pickle

# import third-party modules
import optuna

# import your own module
from ..abstractmodel import AbstractModel


def init_trial(study: optuna.study.Study, model: AbstractModel):
    """
    TODO: Add documentation
    """
    new_trial = study.ask()
    model.parametrize_from_trial(new_trial)

    # Create folder for trial
    study_dir = pathlib.Path(study.user_attrs["study_dir"])
    trial_dir = study_dir / f"trial_{new_trial.number}"
    os.mkdir(trial_dir)

    # Pickle trial
    file_path = study_dir / "current_trial.pickle"
    with open(file_path, "wb") as file:
        pickle.dump(new_trial, file)
