# -*- coding: utf-8 -*-

# import built-in module
import json
import pathlib
import os
import shutil
from typing import Iterator, Tuple, List, Union, Dict, Literal
import warnings
from collections.abc import Iterable
import pickle
import time
import datetime
import inspect

# import third-party modules
import numpy as np
import sklearn.utils
import optuna
import tensorflow as tf
from codecarbon import OfflineEmissionsTracker
import dill

# import your own module
from ..abstractmodel import AbstractModel


def setup_model_study(base_dir, name,
                      sampler: optuna.samplers.BaseSampler,
                      train_data: List[np.ndarray],
                      test_data: List[np.ndarray],
                      nb_trials: int,
                      nb_gpus: int,
                      model: AbstractModel):

    # TODO: Add argument to enable overwriting an existing model study.

    # TODO: Account for requirement to measure the energy for each CV fold

    if len(train_data) != len(test_data):
        raise ValueError(f"Mismatch in the number of elements of train_data "
                         f"({len(train_data)}) and test_data "
                         f"({len(test_data)}). Each element correspond to an"
                         f"outer fold for the nested cross-validation, so the "
                         f"number of elements should be equivalent.")
    nb_outer_folds = len(train_data)
    nb_inner_folds = nb_outer_folds - 1


    # Create base directory
    base_dir = pathlib.Path(base_dir)
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    study_storage = f"sqlite:///{base_dir.resolve()}/study_storage.db"

    # Save data
    data_dir = base_dir / "data"
    os.mkdir(data_dir)
    for fold, train_data in enumerate(train_data):
        np.savez(data_dir / f"fold_{fold}_train.npz", train_data)
    for fold, test_data in enumerate(test_data):
        np.savez(data_dir / f"fold_{fold}_test.npz", test_data)

    datetime_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # One study per outer fold
    for outer_fold in range(nb_outer_folds):
        outer_fold_dir = base_dir / f"outer_fold_{outer_fold}"
        os.mkdir(outer_fold_dir)

        # Pickle a distinct sampler for every outer fold because the sampler
        # state is not stored in the study storage database.
        sampler_path = outer_fold_dir / "sampler.pickle"
        with open(sampler_path, "wb") as f:
            pickle.dump(sampler, f)

        study_name = name + "_outerfold_" + str(outer_fold)
        study = optuna.create_study(storage=study_storage,
                                    study_name=study_name)

        study.set_user_attr("sampler", type(sampler).__name__)
        study.set_user_attr("outerfold", outer_fold)
        study.set_user_attr("study_dir", str(outer_fold_dir.resolve()))
        study.set_user_attr("data_dir", str(data_dir.resolve()))
        study.set_user_attr("sampler_path", str(sampler_path.resolve()))

        run_study_sh_lines = ["#!/bin/bash",
                              f"echo 'Run study - {name} - outer fold {outer_fold}'",
                              "",
                              f"for i in {{0..{nb_trials - 1}}}",
                              "do",
                              "  bash run_trial.sh",
                              "done",
                              "",
                              "echo 'Study complete.'"]
        with open(outer_fold_dir / "run_study.sh", "w") as f:
            f.writelines([line + "\n" for line in run_study_sh_lines])

        obj1 = "test"
        obj2 = "test"
        ts_nb_jobs = nb_gpus + 1
        run_trial_sh_lines = ["#!/bin/bash",
                              f"echo 'Run trial - {name} - outer fold {outer_fold}'",
                              "",
                              f"export PYTHONPATH='{os.environ['PYTHONPATH']}'",
                              "",
                              f"python _init_trial.py",
                              "",
                              "# Max. 2 GPU jobs at a time, + 1 CPU job for remote testbench",
                              f"ts -S {ts_nb_jobs}",
                              "ts --set_gpu_free_perc 80",
                              "ts -C",
                              "",
                              "# Queue the training/testing jobs",
                              "testbench_job=$(ts -G 0 python _get_hardware_metrics.py)",
                              f"for i in {{0..{nb_inner_folds - 1}}}",
                              "do",
                              "  ts -G 1 python ../train_test_fold.py -f ${i}",
                              "done",
                              "",
                              "# Wait for all jobs to be complete.",
                              "ts -w $testbench_job",
                              "ts -w",
                              "",
                              "# Complete trial",
                              f"python ../report_trial.py -t current_trial.pickle -n {name} -s sqlite:///study_storage.db --obj1 {obj1} --obj2 {obj2}",
                              "",
                              "echo 'Trial complete.'"]
        with open(outer_fold_dir / "run_trial.sh", "w") as file:
            file.writelines([line + "\n" for line in run_trial_sh_lines])

        model_class_path = pathlib.Path(inspect.getfile(model.__class__))
        model_class_name = model.__class__.__name__
        init_trial_py_lines = ["# -*- coding: utf-8 -*-",
                               "",
                               f"# NOTE: This script was automatically generated by setup_model_study() on {datetime_str}.",
                               "",
                               "# import built-in module",
                               "import pickle",
                               "import sys",
                               "import importlib.util"
                               "",
                               "# import third-party modules",
                               "import optuna",
                               "",
                               "# import your own module",
                               "from brainmepnas.modelstudy.init_trial import init_trial",
                               "",
                               "# import the AbstractModel implementation",
                               f"spec = importlib.util.spec_from_file_location(\"model\", r\"{model_class_path.resolve()}\")",
                               "model_module = importlib.util.module_from_spec(spec)",
                               "sys.modules[\"model\"] = model_module",
                               "spec.loader.exec_module(model_module)",
                               "",
                               f"sampler = pickle.load(open(r\"{sampler_path.resolve()}\", \"rb\"))",
                               f"study = optuna.load_study(storage=r\"{study_storage}\",",
                               f"                          study_name=\"{study_name}\",",
                               f"                          sampler=sampler)",
                               f"init_trial(study, model_module.{model_class_name}())"]
        with open(outer_fold_dir / "_init_trial.py", "w") as f:
            f.writelines([line + "\n" for line in init_trial_py_lines])

        get_hardware_metrics_py_lines = ["# -*- coding: utf-8 -*-",
                                         "",
                                         f"# NOTE: This script was automatically generated by setup_model_study() on {datetime_str}.",
                                         "",
                                         "# import built-in module",
                                         "import pickle",
                                         "import sys",
                                         "import importlib.util",
                                         "",
                                         "# import third-party modules",
                                         "import optuna",
                                         "",
                                         "# import your own module",
                                         "from brainmepnas.modelstudy.get_hardware_metrics import get_hardware_metrics",
                                         "",
                                         "# import the AbstractModel implementation",
                                         f"spec = importlib.util.spec_from_file_location(\"model\", r\"{model_class_path.resolve()}\")",
                                         "model_module = importlib.util.module_from_spec(spec)",
                                         "sys.modules[\"model\"] = model_module",
                                         "spec.loader.exec_module(model_module)",
                                         "",
                                         "if __name__ == \"__main__\":",
                                         "    trial = pickle.load(open(\"current_trial.pickle\", \"rb\"))",
                                         f"    model = model_module.{model_class_name}()",
                                         "    get_hardware_metrics(model, trial)"]
        with open(outer_fold_dir / "_get_hardware_metrics.py", "w") as f:
            f.writelines([line + "\n" for line in get_hardware_metrics_py_lines])


