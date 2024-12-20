# -*- coding: utf-8 -*-

# import built-in module
import abc
import pathlib
import os
import pickle
import tempfile
from typing import Callable, Dict, List, Any, Literal, Optional
import datetime
import time
import warnings

# import third-party modules
import optuna
import pandas as pd
import click
import codecarbon

# import your own module
from brainmepnas import AccuracyMetrics, CombinedMetrics, HardwareMetrics


class AbstractModelStudy(abc.ABC):
    """
    Abstract implementation of a model study.

    To create a model study,
        1) In a new .py file, create a class with inherits from this class.
        2) Define all class attributes
        3) Implement the following class methods:
            - _sample_search_space
            - _get_accuracy_metrics
            - _get_hardware_metrics
            - _get_combined_metrics
        4) To enable command-line operation, add the following at the bottom of
        the file:
            ```
            class MyModelStudy(AbstractModelStudy):
                [...]


            if __name__ == "__main__":
                MyModelStudy.cli_entry_point()
            ```

    To run a model study, assuming the model study has the name
    "my_model_study", the base directory "base_dir" and is defined in the file
    mymodelstudy.py:
        1) Test your class methods implementations
            ```
            python mymodelstudy.py self-test
            ```
        2) Set up the inner loops files
            ```
            python mymodelstudy.py setup-inner-loops
            ```
        3) Run all inner loops
            ```
            ./base_dir/run_all_inner_loops.sh
            ```
        3.1) Alternatively, you can run each inner loop individually
            ```
            ./base_dir/outer_fold_0/run_inner_loop.sh
            ./base_dir/outer_fold_1/run_inner_loop.sh
            ...
            ```
        4) (Optional) Monitor the progress of each study with optuna-dashboard
            ```
            optuna-dashboard sqlite:///base_dir/study_storage.db
            ```
        5) Set up the outer loop files based on the Pareto sets obtained from
        the inner loops
            ```
            python mymodelstudy.py setup-outer-loop
            ```
        6) Fully-train the Pareto sets from the inner loops and test them on
        unseen data
            ```
            ./base_dir/run_outer_loop.sh
            ```
        6.1) Alternatively, you can fully-train and test the Pareto sets from
        each inner loop separately
            ```
            ./base_dir/outer_fold_0/process_pareto_set.sh
            ./base_dir/outer_fold_1/process_pareto_set.sh
            ...
            ```

    Class attributes
    ----------------
    NAME: str
        Unique name for the model study.
    SAMPLER: optuna.samplers.BaseSampler
        Optuna sampler object (search algorithm).
    BASE_DIR: pathlib.Path
        Directory to store all model study files. This directory should not
        exist beforehand, it will be created when the model study setups
        itself.
    N_OUTER_FOLDS: int
        Number of cross-validation folds to validate the generalization of the
        optimization process.
    N_INNER_FOLDS: int
        Number of cross-validation folds for the validation inside a single
        trial.
    N_TRIALS: int
        Number of different models to try in each inner loop.
    THIS_FILE: pathlib.Path
        Path to the implementation file. This should always be set to __file__:
        ```
        THIS_FILE = __file__
        ```
    OBJ_1_METRIC: str
        Metric to use as the first objective. This should be an attribute of
        AccuracyMetrics, HardwareMetrics, or CombinedMetrics.
    OBJ_1_SCALING: Callable
        Function to use to normalize the first objective. This does not have to
        be a perfect normalization, it is however good to keep the objectives
        roughly in the [0, 1] range.
    OBJ_1_DIRECTION: str
        Whether the first objective should be minimized or maximized.
    OBJ_2_METRIC: str
        Metric to use as the second objective. This should be an attribute of
        AccuracyMetrics, HardwareMetrics, or CombinedMetrics.
    OBJ_2_SCALING: Callable
        Function to use to normalize the second objective. This does not have
        to be a perfect normalization, it is however good to keep the
        objectives roughly in the [0, 1] range.
    OBJ_2_DIRECTION: str
        Whether the second objective should be minimized or maximized.
    N_PARALLEL_GPU_JOBS: int
        Number of parallel GPU jobs to run. It is recommended to set it to the
        number of available GPUs.
    N_PARALLEL_CPU_JOBS: int
        Number of parallel CPU jobs to run.
        It is recommended to set it to >= 1.
    GET_ACCURACY_METRICS_USE_GPU: bool
        Whether get_accuracy_metrics should be called with a GPU.
    GET_HARDWARE_METRICS_USE_GPU: bool
        Whether get_hardware_metrics should be called with a GPU.
    GET_HARDWARE_METRICS_CALL: Literal["once", "per_inner_fold", "never"]
        When to call the get_hardware_metrics method:
            "never": The metrics used as objectives are not attributed of
                     HardwareMetrics or CombinedMetrics, so there is no need to
                     call get_hardware_metrics.
            "once": The HardwareMetrics should be computed once per trial. Use
                    this when the hardware metrics do not depend on the
                    different training folds. For example, in the case of a
                    neural network, the energy and latency do not depend on the
                    weight values, rather only on the structure.
            "per_inner_fold": The HardwareMetrics should be computed once per
                              inner fold. Use this when the hardware metrics
                              depend on the different training folds. For
                              example, in the case of a random forest
                              classifier, the depth and structure of each tree
                              depends on the training data, which means that
                              the energy and latency will also depend on the
                              training data.
    """
    # TODO: Consider adding methods to run the whole studies in Python, with/
    #       without multiprocessing.
    # TODO: Support an arbitrary amount of objectives (1 to ...)

    # ----------------------------------------------
    # - Class attributes to be defined by the user -
    # ----------------------------------------------

    # Note: All class attributes should be defined by the user. This is checked
    #  in the __init_subclass__() method.

    # Model study
    NAME: str
    SAMPLER: optuna.samplers.BaseSampler
    BASE_DIR: pathlib.Path
    N_INNER_FOLDS: int
    N_OUTER_FOLDS: int
    N_TRIALS: int
    THIS_FILE: pathlib.Path

    # Objectives
    OBJ_1_METRIC: str
    OBJ_1_SCALING: Callable
    OBJ_1_DIRECTION: str
    OBJ_2_METRIC: str
    OBJ_2_SCALING: Callable
    OBJ_2_DIRECTION: str

    # Jobs queue
    N_PARALLEL_GPU_JOBS: int
    N_PARALLEL_CPU_JOBS: int
    GET_ACCURACY_METRICS_USE_GPU: bool
    GET_HARDWARE_METRICS_USE_GPU: bool
    GET_HARDWARE_METRICS_CALL: Literal["once", "per_inner_fold", "never"]

    # ----------------------------------------------
    # - Abstract methods to be implemented by user -
    # ----------------------------------------------

    # Note: We manually raise NotImplementedError instead of using the
    #  abc.abstractmethod decorator because the latter does not prevent an
    #  abstract class method from being called.

    @classmethod
    def _sample_search_space(cls, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample the parameters to optimize from the search space.

        This is called once in init_trial(), and most probably in user
        implementations of get_hardware_metrics and get_accuracy_metrics.

        Parameters
        ----------
        trial: optuna.Trial
            Optuna trial object.

        Returns
        -------
        params: Dict[str, Any]
            Dictionary of parameter names and their values.
        """
        raise NotImplementedError

    @classmethod
    def _get_accuracy_metrics(cls, trial: optuna.Trial,
                              trial_dir: pathlib.Path,
                              loop: Literal["inner", "outer"],
                              outer_fold: int,
                              inner_fold: Optional[int] = None) -> AccuracyMetrics:
        """
        Calculate the accuracy metrics for a given trial and inner/outer loop
        iteration. This private method is called by the public method
        get_accuracy_metrics(), which additionally pickles the AccuracyMetrics
        object and tracks estimated carbon emissions.

        If your implementation benefits from a GPU, set
        GET_ACCURACY_METRICS_USE_GPU=True.

        This is called once per inner fold.

        Note: to share information between get_accuracy_metrics() and
        get_hardware_metrics(), save information to file using pickle, csv,
        json, ...

        Parameters
        ----------
        trial: optuna.Trial
            Optuna trial object.
        loop: Literal["inner", "outer"]
            Whether we are in the inner or outer loop of the nested
            cross-validation. In the inner loop, the outer fold is excluded
            from the data and the inner fold is used for testing. In the outer
            loop, the outer fold is used for testing and the remaining data is
            used for training.
        outer_fold: int
            Outer fold.
        inner_fold: int, optional
            Inner fold.


        Returns
        -------
        accuracy_metrics: AccuracyMetrics
            AccuracyMetrics object.
        """
        raise NotImplementedError

    @classmethod
    def _get_hardware_metrics(cls, trial: optuna.Trial,
                              trial_dir: pathlib.Path,
                              loop: Literal["inner", "outer"],
                              outer_fold: int,
                              inner_fold: Optional[int] = None) -> HardwareMetrics:
        """
        Calculate the hardware metrics for a given trial and inner/outer loop
        iteration. This private method is called by the public method
        get_hardware_metrics(), which additionally pickles the HardwareMetrics
        object and tracks estimated carbon emissions.

        If your implementation benefits from a GPU, set
        GET_HARDWARE_METRICS_USE_GPU=True.

        This is called depending on the value of GET_HARDWARE_METRICS_CALL:
            "once": called once, independently of inner_fold. If
                    GET_ACCURACY_METRICS_CALL=="once", get_hardware_metrics()
                    is called once after the execution of
                    get_accuracy_metrics() is complete.
            "per_inner_fold": called once per inner_fold. If
                              GET_ACCURACY_METRICS_CALL=="per fold",
                              get_hardware_metrics() is called for each fold
                              after the execution of the corresponding
                              get_accuracy_metrics() is complete.
            "never": not called.

        Note: to share information between get_accuracy_metrics() and
        get_hardware_metrics(), save information to file using pickle, csv,
        json, ...

        Parameters
        ----------
        trial: optuna.Trial
            Optuna trial object.
        loop: Literal["inner", "outer"]
            Whether we are in the inner or outer loop of the nested
            cross-validation. In the inner loop, the outer fold is excluded
            from the data and the inner fold is used for testing. In the outer
            loop, the outer fold is used for testing and the remaining data is
            used for training.
        outer_fold: int
            Outer fold.
        inner_fold: int, optional
            Inner fold.

        Returns
        -------
        hardware_metrics: HardwareMetrics
            HardwareMetrics object.
        """
        raise NotImplementedError

    @classmethod
    def _get_combined_metrics(cls, accuracy_metrics: AccuracyMetrics,
                              hardware_metrics: HardwareMetrics) -> CombinedMetrics:
        """
        Calculate the combined metrics from provided accuracy and hardware
        metrics.

        This function is called by the public get_combined_metrics(),
        which additionally pickles the HardwareMetrics object.

        This is called after processing all folds.

        Parameters
        ----------
        accuracy_metrics: AccuracyMetrics
            AccuracyMetrics object.
        hardware_metrics: HardwareMetrics
            HardwareMetrics object

        Returns
        -------
        combined_metrics: CombinedMetrics
            CombinedMetrics object.
        """
        raise NotImplementedError

    # ------------------
    # - Public methods -
    # ------------------

    @classmethod
    def self_test(cls):
        """
        Perform a self-test of the model study.

        This tests specifically that the user implementation defines all
        required attributes and that all methods run without errors.

        No exception is raised if a step of the test fails. All information is
        provided in warnings.
        """
        # Check that all attributes are defined.
        # Adapted from https://stackoverflow.com/a/55544173
        required_class_variables = ["NAME", "SAMPLER", "BASE_DIR",
                                    "N_OUTER_FOLDS", "N_INNER_FOLDS",
                                    "N_TRIALS", "THIS_FILE", "OBJ_1_METRIC",
                                    "OBJ_1_SCALING", "OBJ_1_DIRECTION",
                                    "OBJ_2_METRIC", "OBJ_2_SCALING",
                                    "OBJ_2_DIRECTION", "N_PARALLEL_GPU_JOBS",
                                    "N_PARALLEL_CPU_JOBS",
                                    "GET_ACCURACY_METRICS_USE_GPU",
                                    "GET_HARDWARE_METRICS_CALL",
                                    "GET_HARDWARE_METRICS_USE_GPU"]
        for var in required_class_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(f"Class attribute '{var}' is "
                                          f"missing.")

        # Create dummy study and dummy trial
        study = optuna.create_study()
        study.set_user_attr("outer_fold", 0)
        trial = study.ask()
        tempdir = tempfile.TemporaryDirectory()
        trial.set_user_attr("trial_dir", tempdir.name)
        trial_dir = pathlib.Path(tempdir.name)

        # _sample_search_space
        try:
            _ = cls._sample_search_space(trial)
        except Exception as e:
            print(f"Exception in _sample_search_space():")
            raise e

        # _get_accuracy_metrics, inner loop
        try:
            am = cls._get_accuracy_metrics(trial, trial_dir, "inner", 0, 1)
        except Exception as e:
            print(f"Exception in _get_accuracy_metrics() with loop='inner':")
            raise e
        else:
            if not isinstance(am, AccuracyMetrics):
                raise TypeError(f"_get_accuracy_metrics() with loop='inner' "
                                f"returned unexpected type: {type(am)}")

        # _get_accuracy_metrics, outer loop
        try:
            am = cls._get_accuracy_metrics(trial, trial_dir, "outer", 0)
        except Exception as e:
            print(
                f"Exception in _get_accuracy_metrics() with loop='outer':")
            raise e
        else:
            if not isinstance(am, AccuracyMetrics):
                raise TypeError(f"_get_accuracy_metrics() with loop='outer' "
                                f"returned unexpected type: {type(am)}")

        # _get_hardware_metrics, inner loop
        try:
            hm = cls._get_hardware_metrics(trial, trial_dir, "inner", 0, 1)
        except Exception as e:
            print(f"Exception in _get_hardware_metrics() with loop='inner':")
            raise e
        else:
            if not isinstance(hm, HardwareMetrics):
                raise TypeError(f"_get_hardware_metrics() with loop='inner' "
                                f"returned unexpected type: {type(hm)}")

        # _get_hardware_metrics, outer loop
        try:
            hm = cls._get_hardware_metrics(trial, trial_dir, "inner", 0)
        except Exception as e:
            print(
                f"Exception in _get_hardware_metrics() with loop='outer':")
            raise e
        else:
            if not isinstance(hm, HardwareMetrics):
                raise TypeError(
                    f"_get_hardware_metrics() with loop='outer' returned "
                    f"unexpected type: {type(hm)}")

        # _get_combined_metrics
        try:
            cm = cls._get_combined_metrics(am, hm)
        except Exception as e:
            print(f"Exception in _get_combined_metrics():")
            raise e
        else:
            if not isinstance(cm, CombinedMetrics):
                raise TypeError(f"_get_combined_metrics() returned unexpected "
                                f"type: {type(cm)}")

        tempdir.cleanup()

    @classmethod
    def setup_inner_loops(cls):
        """
        Set up the inner loops for a model study.

        A model study consists of an outer and an inner loop. The inner loop
        is represented by an Optuna Study object and contains many trials. The
        output of a single inner loop is a Pareto Set, which has been trained
        and tested on a subset of the data for inner loop 1. The outer loop
        consists in evaluating the different Pareto sets on unseen data.

        Example:
            - Desired number of outer folds is N_OUTER_FOLDS, which means that
            the dataset is split in N_OUTER_FOLDS.
            - There is one inner loop per outer fold, where each inner loop is
            performed on the whole dataset excluding one fold. For a given
            inner loop, (N_OUTER_FOLDS-1) folds are made available.
            - In each trial of an inner loop, a cross-validation process is
            performed with N_INNER_FOLDS folds: N_INNER_FOLDS models are
            trained, each with leaving out one of the folds for testing. The
            trial performance is the mean performance of all trials.

        All data related to the model study are stored in BASE_DIR. All studies
        are placed in study_storage.db. Each study has a folder where scripts
        and execution traces are stored.

        Raises
        ------
        FileExistsError
            if the specified BASE_DIR already exists.
        """

        # Create base directory
        if not os.path.isdir(cls.BASE_DIR):
            os.mkdir(cls.BASE_DIR)
        else:
            raise FileExistsError(f"Base directory {cls.BASE_DIR} already "
                                  f"exists. Interrupting setup to prevent "
                                  f"undesired overwrite of existing data.")

        study_storage_url = f"sqlite:///{cls.BASE_DIR.resolve()}/study_storage.db"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        # One study per outer fold
        run_inner_loop_files = []
        for outer_fold in range(cls.N_OUTER_FOLDS):
            outer_fold_dir = cls.BASE_DIR / f"outer_fold_{outer_fold}"
            os.mkdir(outer_fold_dir)

            study_name = cls.NAME + "_outer_fold_" + str(outer_fold)
            study = optuna.create_study(storage=study_storage,
                                        study_name=study_name,
                                        directions=[cls.OBJ_1_DIRECTION,
                                                    cls.OBJ_2_DIRECTION])

            # Note: Sampler is pickled because it is not stored in
            # study_storage.
            sampler_path = outer_fold_dir / "sampler.pickle"
            with open(sampler_path, "wb") as f:
                pickle.dump(cls.SAMPLER, f)

            study.set_user_attr("sampler", type(cls.SAMPLER).__name__)
            study.set_user_attr("outer_fold", outer_fold)
            study.set_user_attr("study_dir", str(outer_fold_dir.resolve()))
            study.set_user_attr("sampler_path", str(sampler_path.resolve()))

            run_trial_sh_path = cls._create_run_trial_sh(outer_fold_dir,
                                                         study_storage_url,
                                                         study_name,
                                                         sampler_path)
            run_inner_loop_sh_path = cls._create_run_inner_loop_sh(outer_fold_dir,
                                                                   run_trial_sh_path)
            run_inner_loop_files.append(run_inner_loop_sh_path)
        cls._create_run_all_inner_loops_sh(cls.BASE_DIR, run_inner_loop_files)

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

    @classmethod
    def setup_outer_loop(cls):
        """
        Set up the outer loop for a model study.

        A model study consists of an outer and an inner loop. The inner loop
        is represented by an Optuna Study object and contains many trials. The
        output of a single inner loop is a Pareto Set, which has been trained
        and tested on a subset of the data for inner loop 1. The outer loop
        consists in evaluating the different Pareto sets on unseen data.

        All data related to the model study are stored in BASE_DIR. All studies
        are placed in study_storage.db. Each study has a folder where scripts
        and execution traces are stored.
        """
        study_storage_url = (f"sqlite:///{cls.BASE_DIR.resolve()}/"
                             f"study_storage.db")
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        run_outer_loop_iteration_files = []
        for outer_fold in range(cls.N_OUTER_FOLDS):
            study_name = cls.NAME + "_outer_fold_" + str(outer_fold)
            study = optuna.load_study(storage=study_storage,
                                      study_name=study_name)
            n_trials = len(study.trials)
            if n_trials < cls.N_TRIALS:
                warnings.warn(f"Study {study_name} has {n_trials}, which is "
                              f"less that the expected "
                              f"cls.N_TRIALS={cls.N_TRIALS}. Are you sure the "
                              f"inner loop was fully completed?")
            study_dir = pathlib.Path(study.user_attrs["study_dir"])
            process_pareto_set_sh_path = (
                cls._create_process_pareto_set_sh(study_dir,
                                                  study.best_trials))
            run_outer_loop_iteration_files.append(
                process_pareto_set_sh_path)

        cls._create_run_outer_loop_sh(cls.BASE_DIR,
                                      run_outer_loop_iteration_files)

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

    @classmethod
    def init_trial(cls, study: optuna.Study) -> optuna.Trial:
        """
        Initialize a new trial.

        Trial is obtained from the study object and parameters are sampled from
        the search space. A new folder is created to store trial data. Finally,
        the trial is pickled to be accessible by other processes.

        Parameters
        ----------
        study: optuna.Study
            Current study.

        Returns
        -------
        trial: optuna.Trial
        """
        start_time = time.time()
        new_trial = study.ask()

        # sampled parameters are discarded because they are fixed in the trial
        # when they are sampled once. Here, we only want to fix the values in
        # the trial before pickling it.
        _ = cls._sample_search_space(new_trial)

        # Create folder for trial
        study_dir = pathlib.Path(study.user_attrs["study_dir"])
        trial_dir = study_dir / f"trial_{new_trial.number}"
        new_trial.set_user_attr("trial_dir", str(trial_dir.resolve()))
        new_trial.set_user_attr("outer_fold",
                                str(study.user_attrs["outer_fold"]))
        os.mkdir(trial_dir)

        # Pickle trial
        file_path = study_dir / "current_trial.pickle"
        with open(file_path, "wb") as file:
            pickle.dump(new_trial, file)

        duration = time.time() - start_time
        new_trial.set_user_attr("init_trial_duration", duration)

        return new_trial

    @classmethod
    def get_accuracy_metrics(cls, trial: optuna.Trial,
                             loop: Literal["inner", "outer"],
                             inner_fold: Optional[int] = None) -> AccuracyMetrics:
        """
        Calculate and save accuracy metrics for a specific trial and
        inner/outer loop iteration to the trial directory as
        inner_fold_{inner_fold}_accuracy_metrics.pickle or
        outer_fold_{outer_fold}_accuracy_metrics.pickle. Carbon emissions are
        tracked and saved to emissions.csv.

        This is called depending on the value of GET_ACCURACY_METRICS_CALL:
            "once": called once, independently of inner_fold. If
                    GET_HARDWARE_METRICS_CALL=="once", get_hardware_metrics()
                    is called once after the execution of
                    get_accuracy_metrics() is complete.
            "per_inner_fold": called once per inner_fold. If
                              GET_HARDWARE_METRICS_CALL=="per_inner_fold",
                              get_hardware_metrics() is called for each fold
                              after the execution of the corresponding
                              get_accuracy_metrics() is complete.
            "never": not called.

        Parameters
        ----------
        trial: optuna.Trial
            Optuna trial object.
        loop: Literal["inner", "outer"]
            Whether we are in the inner or outer loop of the nested
            cross-validation. In the inner loop, the outer fold is excluded
            from the data and the inner fold is used for testing. In the outer
            loop, the outer fold is used for testing and the remaining data is
            used for training.
        inner_fold: int, optional
            Inner fold.

        Returns
        -------
        accuracy_metrics: AccuracyMetrics
            AccuracyMetrics object.
        """
        start_time = time.time()

        outer_fold = int(trial.user_attrs["outer_fold"])
        trial_dir = pathlib.Path(trial.user_attrs["trial_dir"])

        # Verify parameters
        if loop not in ["inner", "outer"]:
            raise ValueError("loop must be either 'inner' or 'outer'.")
        if inner_fold is not None:
            if not (0 <= inner_fold < cls.N_INNER_FOLDS):
                raise ValueError("inner_fold must be an integer between 0 and "
                                 "N_FOLDS.")

        # Prepare description
        if loop == "inner":
            if inner_fold is None:
                description = f"inner_fold_all_accuracy_metrics"
            else:
                description = f"inner_fold_{inner_fold}_accuracy_metrics"
        else:
            description = f"outer_fold_{outer_fold}_accuracy_metrics"

        # Start carbon tracking
        tracker = codecarbon.OfflineEmissionsTracker(country_iso_code="DEU",
                                                     project_name=description,
                                                     tracking_mode="process",
                                                     output_dir=trial_dir,
                                                     log_level="WARNING")
        tracker.start()

        # Call user implementation
        am = cls._get_accuracy_metrics(trial, trial_dir, loop, outer_fold,
                                       inner_fold)

        # End carbon tracking
        tracker.stop()

        # Pickle AccuracyMetrics object
        am_path = trial_dir / (description + ".pickle")
        pickle.dump(am, open(am_path, "wb"))

        # Save duration to trial attributes
        duration = time.time() - start_time
        trial.set_user_attr(f"{description}_duration", duration)
        return am

    @classmethod
    def get_hardware_metrics(cls, trial: optuna.Trial,
                             loop: Literal["inner", "outer"],
                             inner_fold: Optional[int] = None) -> HardwareMetrics:
        """
        Calculate and save hardware metrics for a specific trial and
        inner/outer loop iteration to the trial directory as
        inner_fold_{inner_fold}_hardware_metrics.pickle or
        outer_fold_{outer_fold}_hardware_metrics.pickle. Carbon emissions are
        tracked and saved to emissions.csv.

        This is called depending on the value of GET_HARDWARE_METRICS_CALL:
            "once": called once, independently of inner_fold. If
                    GET_ACCURACY_METRICS_CALL=="once", get_hardware_metrics()
                    is called once after the execution of
                    get_accuracy_metrics() is complete.
            "per_inner_fold": called once per inner_fold. If
                              GET_ACCURACY_METRICS_CALL=="per fold",
                              get_hardware_metrics() is called for each fold
                              after the execution of the corresponding
                              get_accuracy_metrics() is complete.
            "never": not called.

        Parameters
        ----------
        trial: optuna.Trial
            Optuna trial object.
        loop: Literal["inner", "outer"]
            Whether we are in the inner or outer loop of the nested
            cross-validation. In the inner loop, the outer fold is excluded
            from the data and the inner fold is used for testing. In the outer
            loop, the outer fold is used for testing and the remaining data is
            used for training.
        inner_fold: int, optional
            Inner fold.

        Returns
        -------
        hardware_metrics: HardwareMetrics
            HardwareMetrics object.
        """
        start_time = time.time()

        outer_fold = int(trial.user_attrs["outer_fold"])
        trial_dir = pathlib.Path(trial.user_attrs["trial_dir"])

        # Verify parameters
        if loop not in ["inner", "outer"]:
            raise ValueError("loop must be either 'inner' or 'outer'.")
        if inner_fold is not None:
            if not (0 <= inner_fold < cls.N_INNER_FOLDS):
                raise ValueError("inner_fold must be an integer between 0 and "
                                 "N_FOLDS.")

        # Prepare description
        if loop == "inner":
            if inner_fold is None:
                description = f"inner_fold_all_hardware_metrics"
            else:
                description = f"inner_fold_{inner_fold}_hardware_metrics"
        else:
            description = f"outer_fold_{outer_fold}_hardware_metrics"

        # Start carbon tracking
        tracker = codecarbon.OfflineEmissionsTracker(country_iso_code="DEU",
                                                     project_name=description,
                                                     tracking_mode="process",
                                                     output_dir=trial_dir,
                                                     log_level="WARNING")
        tracker.start()

        # Call user implementation
        hm = cls._get_hardware_metrics(trial, trial_dir, loop, outer_fold,
                                       inner_fold)

        # End carbon tracking
        tracker.stop()

        # Pickle HardwareMetrics object
        hm_path = trial_dir / (description + ".pickle")
        pickle.dump(hm, open(hm_path, "wb"))

        # Save duration to trial attributes
        duration = time.time() - start_time
        trial.set_user_attr(f"{description}_duration", duration)
        return hm

    @classmethod
    def get_combined_metrics(cls, accuracy_metrics: AccuracyMetrics,
                             hardware_metrics: HardwareMetrics,
                             trial: optuna.Trial,
                             loop: Literal["inner", "outer"],
                             inner_fold: Optional[int] = None) -> CombinedMetrics:
        """
        Calculate and save combined metrics from accuracy and hardware metrics
        to the trial directory as
        inner_fold_{inner_fold}_combined_metrics.pickle or
        outer_fold_{outer_fold}_combined_metrics.pickle.

        This is called after processing all folds.

        Parameters
        ----------
        accuracy_metrics: AccuracyMetrics
            AccuracyMetrics object.
        hardware_metrics: HardwareMetrics
            HardwareMetrics object
        trial: optuna.Trial
            Optuna trial object.
        loop: Literal["inner", "outer"]
            Whether we are in the inner or outer loop of the nested
            cross-validation. In the inner loop, the outer fold is excluded
            from the data and the inner fold is used for testing. In the outer
            loop, the outer fold is used for testing and the remaining data is
            used for training.
        inner_fold: int, optional
            Inner fold.

        Returns
        -------
        combined_metrics: CombinedMetrics
            CombinedMetrics object.
        """
        start_time = time.time()

        outer_fold = trial.user_attrs["outer_fold"]
        trial_dir = pathlib.Path(trial.user_attrs["trial_dir"])

        # Verify parameters
        if loop not in ["inner", "outer"]:
            raise ValueError("loop must be either 'inner' or 'outer'.")
        if inner_fold is not None:
            if not (0 <= inner_fold < cls.N_INNER_FOLDS):
                raise ValueError("inner_fold must be an integer between 0 and "
                                 "N_FOLDS.")

        # Prepare description
        if loop == "inner":
            if inner_fold is None:
                description = f"inner_fold_all_combined_metrics"
            else:
                description = f"inner_fold_{inner_fold}_combined_metrics"
        else:
            description = f"outer_fold_{outer_fold}_combined_metrics"

        # Start carbon tracking
        tracker = codecarbon.OfflineEmissionsTracker(country_iso_code="DEU",
                                                     project_name=description,
                                                     tracking_mode="process",
                                                     output_dir=trial_dir,
                                                     log_level="WARNING")
        tracker.start()

        # Call user implementation
        cm = cls._get_combined_metrics(accuracy_metrics, hardware_metrics)

        # End carbon tracking
        tracker.stop()

        # Pickle CombinedMetrics object
        cm_path = trial_dir / (description + ".pickle")
        pickle.dump(cm, open(cm_path, "wb"))

        # Save duration to trial attributes
        duration = time.time() - start_time
        trial.set_user_attr(f"{description}_duration", duration)
        return cm

    @classmethod
    def complete_trial(cls, trial: optuna.Trial,
                       loop: Literal["inner", "outer"]):
        """
        Complete the given trial by compiling the produced metrics and
        exporting them to a .csv file.

        If loop=inner, the objectives are extracted and reported to the study.
        If loop=outer, nothing more is done because the trial was already
        completed.

        Parameters
        ----------
        trial: optuna.Trial
            Trial object.
        loop: Literal["inner", "outer"]
            Whether we are in the inner or outer loop of the nested
            cross-validation. In the inner loop, the outer fold is excluded
            from the data and the inner fold is used for testing. In the outer
            loop, the outer fold is used for testing and the remaining data is
            used for training.
        """
        start_time = time.time()

        outer_fold = int(trial.user_attrs["outer_fold"])
        trial_dir = pathlib.Path(trial.user_attrs["trial_dir"])
        try:
            study = trial.study
        except AttributeError:
            # Manually load study
            study_storage_url = (f"sqlite:///{cls.BASE_DIR.resolve()}/"
                                 f"study_storage.db")
            study_name = cls.NAME + "_outer_fold_" + str(outer_fold)
            study = optuna.load_study(storage=study_storage_url,
                                      study_name=study_name)

        # Verify parameter
        if loop not in ["inner", "outer"]:
            raise ValueError("loop must be either 'inner' or 'outer'.")

        if loop == "inner":

            metrics_dicts = []

            try:
                # For each inner fold, get CombinedMetrics from AccuracyMetrics
                # and HardwareMetrics
                for f in range(cls.N_INNER_FOLDS):
                    am_path = (trial_dir /
                               f"inner_fold_{f}_accuracy_metrics.pickle")
                    am = pickle.load(open(am_path, "rb"))

                    if cls.GET_HARDWARE_METRICS_CALL == "once":
                        hm_path = (trial_dir /
                                   f"inner_fold_all_hardware_metrics.pickle")
                        hm = pickle.load(open(hm_path, "rb"))
                    elif cls.GET_HARDWARE_METRICS_CALL == "per_inner_fold":
                        hm_path = (trial_dir /
                                   f"inner_fold_{f}_hardware_metrics.pickle")
                        hm = pickle.load(open(hm_path, "rb"))
                    else:
                        hm = None

                    if am is not None and hm is not None:
                        cm = cls.get_combined_metrics(am, hm, trial, loop, f)
                    else:
                        cm = None

                    d = {"trial": trial.number,
                         "inner_fold": f}
                    if am is not None:
                        d.update(am.as_dict())
                    if hm is not None:
                        d.update(hm.as_dict())
                    if cm is not None:
                        d.update(cm.as_dict())
                    metrics_dicts.append(d)
            except FileNotFoundError:
                # Fail trial if one of the expected metrics file is absent.
                duration = time.time() - start_time
                trial.set_user_attr(f"complete_trial_duration", duration)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                sampler_path = pathlib.Path(study.user_attrs["sampler_path"])
                pickle.dump(study.sampler, open(sampler_path, "wb"))
                return None

            # Save all metrics to .csv file.
            df = pd.DataFrame.from_records(metrics_dicts)
            csv_file_path = (cls.BASE_DIR / f"outer_fold_{outer_fold}" /
                             f"inner_loop_metrics.csv")
            if csv_file_path.exists():
                df.to_csv(csv_file_path, index=False, mode="a", header=False)
            else:
                df.to_csv(csv_file_path, index=False, mode="w", header=True)

            # Report objectives to trial.
            try:
                obj_1_value = df[cls.OBJ_1_METRIC].mean()
                obj_1_value_scaled = cls.OBJ_1_SCALING(obj_1_value)

                obj_2_value = df[cls.OBJ_2_METRIC].mean()
                obj_2_value_scaled = cls.OBJ_2_SCALING(obj_2_value)
            except KeyError:
                # Fail trial if objective is not found in the metrics.
                duration = time.time() - start_time
                trial.set_user_attr(f"complete_trial_duration", duration)
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                sampler_path = pathlib.Path(study.user_attrs["sampler_path"])
                pickle.dump(study.sampler, open(sampler_path, "wb"))
            else:
                trial.set_user_attr(cls.OBJ_1_METRIC, obj_1_value)
                trial.set_user_attr(cls.OBJ_2_METRIC, obj_2_value)
                duration = time.time() - start_time
                trial.set_user_attr(f"complete_trial_duration", duration)
                study.tell(trial, [obj_1_value_scaled, obj_2_value_scaled])
                sampler_path = pathlib.Path(study.user_attrs["sampler_path"])
                pickle.dump(study.sampler, open(sampler_path, "wb"))
        elif loop == "outer":
            # Get CombinedMetrics from AccuracyMetrics and HardwareMetrics
            am_path = (trial_dir /
                       f"outer_fold_{outer_fold}_accuracy_metrics.pickle")
            am = pickle.load(open(am_path, "rb"))
            if cls.GET_HARDWARE_METRICS_CALL == "once":
                hm_path = trial_dir / f"inner_fold_all_hardware_metrics.pickle"
                hm = pickle.load(open(hm_path, "rb"))
            elif cls.GET_HARDWARE_METRICS_CALL == "per_inner_fold":
                hm_path = (trial_dir /
                           f"outer_fold_{outer_fold}_hardware_metrics.pickle")
                hm = pickle.load(open(hm_path, "rb"))
            else:
                hm = None
            if am is not None and hm is not None:
                cm = cls.get_combined_metrics(am, hm, trial, "outer")
            else:
                cm = None
            d = {"trial": trial.number,
                 "outer_fold": outer_fold}
            if am is not None:
                d.update(am.as_dict())
            if hm is not None:
                d.update(hm.as_dict())
            if cm is not None:
                d.update(cm.as_dict())
            d.update(am.as_dict())
            d.update(hm.as_dict())
            d.update(cm.as_dict())

            # Save all metrics to .csv file.
            df = pd.DataFrame.from_records([d])
            csv_file_path = cls.BASE_DIR / f"outer_loop_metrics.csv"
            if csv_file_path.exists():
                df.to_csv(csv_file_path, index=False, mode="a", header=False)
            else:
                df.to_csv(csv_file_path, index=False, mode="w", header=True)

    def __init__(self):
        """
        This class is not meant to be instantiated. All methods and attributes
        are class methods and class attributes.

        Raises
        ------
        RuntimeError
        """
        raise RuntimeError

    @classmethod
    def _create_run_trial_sh(cls, target_dir: pathlib.Path,
                             study_storage: str, study_name: str,
                             sampler_path: pathlib.Path) -> pathlib.Path:
        """
        Generate run_trial.sh script in target_dir.
        """
        datetime_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        n_total_jobs = cls.N_PARALLEL_CPU_JOBS + cls.N_PARALLEL_GPU_JOBS
        trial_path = target_dir / "current_trial.pickle"

        lines = ["#!/bin/bash",
                 "",
                 "# NOTE: This script was automatically generated by",
                 f"# {cls.__name__}._create_run_trial_sh() on {datetime_str}",
                 "",
                 f"echo 'Run trial'",
                 "",
                 "# Initialize trial",
                 f"python {cls.THIS_FILE} init_trial -u {study_storage} -n {study_name} -s {sampler_path.resolve()}",
                 "",
                 "# Configure task spooler",
                 f"# {cls.N_PARALLEL_CPU_JOBS} CPU jobs + {cls.N_PARALLEL_GPU_JOBS} GPU jobs = {n_total_jobs} total jobs",
                 f"ts -S {n_total_jobs}",
                 "ts -C"]

        if (cls.GET_ACCURACY_METRICS_USE_GPU or
                cls.GET_HARDWARE_METRICS_USE_GPU):
            lines += ["ts --set_gpu_free_perc 80"]

        lines += ["",
                  "# Queue jobs"]

        if cls.GET_ACCURACY_METRICS_USE_GPU:
            get_accuracy_metrics_gpu_option = "-G 1 "
        else:
            get_accuracy_metrics_gpu_option = ""

        if cls.GET_HARDWARE_METRICS_USE_GPU:
            get_hardware_metrics_gpu_option = "-G 1 "
        else:
            get_hardware_metrics_gpu_option = ""
        job_names = []

        # Call once
        if cls.GET_HARDWARE_METRICS_CALL == "once":
            job_names.append(f"job_{len(job_names)}")
            lines += [f"{job_names[-1]}=$(ts {get_hardware_metrics_gpu_option}python {cls.THIS_FILE} get_hardware_metrics -t {trial_path.resolve()} --inner-loop)"]

        # Call per_inner_fold
        for inner_fold in range(cls.N_INNER_FOLDS):
            job_names.append(f"job_{len(job_names)}")
            lines += [f"{job_names[-1]}=$(ts {get_accuracy_metrics_gpu_option}python {cls.THIS_FILE} get_accuracy_metrics -t {trial_path.resolve()} --inner-loop -i {inner_fold})"]

            if cls.GET_HARDWARE_METRICS_CALL == "per_inner_fold":
                job_names.append(f"job_{len(job_names)}")
                # Wait for last get_accuracy_metrics job to complete.
                lines += [f"{job_names[-1]}=$(ts -D ${job_names[-2]} {get_hardware_metrics_gpu_option} python {cls.THIS_FILE} get_hardware_metrics -t {trial_path.resolve()} --inner-loop -i {inner_fold})"]

        # Wait for all jobs to complete
        lines += [""]
        for job_name in job_names:
            lines += [f"ts -w ${job_name}"]

        lines += ["",
                  "# Complete trial",
                  f"python {cls.THIS_FILE} complete_trial -t {trial_path.resolve()} --inner-loop",
                  "",
                  "echo 'Trial complete.'"]

        file_path = target_dir / "run_trial.sh"

        with open(file_path, "w") as f:
            f.writelines([line + "\n" for line in lines])

        return file_path

    @classmethod
    def _create_run_inner_loop_sh(cls, target_dir: pathlib.Path,
                                  run_trial_path: pathlib.Path):
        datetime_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        lines = ["#!/bin/bash",
                 "",
                 "# NOTE: This script was automatically generated by",
                 f"# {cls.__name__}._create_run_inner_loop_sh() on {datetime_str}",
                 "",
                 f"echo 'Run inner loop'",
                 "",
                 f"# {cls.N_TRIALS} trials.",
                 f"for i in {{0..{cls.N_TRIALS-1}}}",
                 "do",
                 f"    bash {run_trial_path.resolve()}",
                 "done",
                 "",
                 "echo 'Inner loop complete.'"]

        file_path = target_dir / "run_inner_loop.sh"

        with open(file_path, "w") as f:
            f.writelines([line + "\n" for line in lines])

        return file_path

    @classmethod
    def _create_run_all_inner_loops_sh(cls, target_dir: pathlib.Path,
                                       run_inner_loop_sh_files: List[pathlib.Path]):
        datetime_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        lines = ["#!/bin/bash",
                 "",
                 "# NOTE: This script was automatically generated by",
                 f"# {cls.__name__}._create_run_all_inner_loops_sh() on {datetime_str}",
                 "",
                 f"echo 'Run all inner loops'",
                 ""]

        for p in run_inner_loop_sh_files:
            lines += [f"bash {p.resolve()}"]

        lines += ["",
                  "echo 'All inner loops complete.'"]

        file_path = target_dir / "run_all_inner_loops.sh"

        with open(file_path, "w") as f:
            f.writelines([line + "\n" for line in lines])

        return file_path

    @classmethod
    def _create_process_pareto_set_sh(cls, target_dir: pathlib.Path,
                                      pareto_set: List[optuna.trial.FrozenTrial]):
        datetime_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        n_total_jobs = cls.N_PARALLEL_CPU_JOBS + cls.N_PARALLEL_GPU_JOBS

        lines = ["#!/bin/bash",
                 "",
                 "# NOTE: This script was automatically generated by",
                 f"# {cls.__name__}._create_process_pareto_set_sh() on {datetime_str}",
                 "",
                 f"echo 'Process Pareto set'",
                 "",
                 "# Configure task spooler",
                 f"# {cls.N_PARALLEL_CPU_JOBS} CPU jobs + {cls.N_PARALLEL_GPU_JOBS} GPU jobs = {n_total_jobs} total jobs",
                 f"ts -S {n_total_jobs}",
                 "ts -C"]

        if (cls.GET_ACCURACY_METRICS_USE_GPU or
                cls.GET_HARDWARE_METRICS_USE_GPU):
            lines += ["ts --set_gpu_free_perc 80"]

        lines += ["",
                  "# Queue jobs"]

        if cls.GET_ACCURACY_METRICS_USE_GPU:
            get_accuracy_metrics_gpu_option = "-G 1 "
        else:
            get_accuracy_metrics_gpu_option = ""

        if cls.GET_HARDWARE_METRICS_USE_GPU:
            get_hardware_metrics_gpu_option = "-G 1 "
        else:
            get_hardware_metrics_gpu_option = ""
        job_names = []

        for trial in pareto_set:
            trial_dir = pathlib.Path(trial.user_attrs["trial_dir"])
            trial_pickle_path = (trial_dir /
                                 f"pareto_set_trial_{trial.number}.pickle")
            pickle.dump(trial, open(trial_pickle_path, "wb"))
            job_names.append(f"job_{len(job_names)}")
            lines += [f"{job_names[-1]}=$(ts {get_accuracy_metrics_gpu_option}python {cls.THIS_FILE} get_accuracy_metrics -t {trial_pickle_path.resolve()} --outer-loop)"]
            if cls.GET_HARDWARE_METRICS_CALL == "per_inner_fold":
                job_names.append(f"job_{len(job_names)}")
                lines += [f"{job_names[-1]}=$(ts -D ${job_names[-2]} {get_hardware_metrics_gpu_option}python {cls.THIS_FILE} get_hardware_metrics -t {trial_pickle_path.resolve()} --outer-loop)"]
            job_names.append(f"job_{len(job_names)}")
            lines += [f"{job_names[-1]}=$(ts -D ${job_names[-2]} python {cls.THIS_FILE} complete_trial -t {trial_pickle_path.resolve()} --outer-loop)"]

        # Wait for all jobs to complete
        lines += [""]
        for job_name in job_names:
            lines += [f"ts -w ${job_name}"]

        lines += ["",
                  "echo 'Processing Pareto Set complete.'"]

        file_path = target_dir / "process_pareto_set.sh"

        with open(file_path, "w") as f:
            f.writelines([line + "\n" for line in lines])

        return file_path

    @classmethod
    def _create_run_outer_loop_sh(cls, target_dir: pathlib.Path,
                                  run_outer_loop_iteration_paths: List[pathlib.Path]):
        datetime_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        lines = ["#!/bin/bash",
                 "",
                 "# NOTE: This script was automatically generated by",
                 f"# {cls.__name__}._create_run_outer_loop_sh() on {datetime_str}",
                 "",
                 f"echo 'Run outer loop'",
                 ""]

        for p in run_outer_loop_iteration_paths:
            lines += [f"bash {p.resolve()}"]

        lines += ["",
                  "echo 'Outer loop complete.'"]

        file_path = target_dir / "run_outer_loop.sh"

        with open(file_path, "w") as f:
            f.writelines([line + "\n" for line in lines])

        return file_path

    # ----------------------------------
    # - Command-line interface methods -
    # ----------------------------------

    @classmethod
    def cli_entry_point(cls):
        """
        Entry point for the command-line interface.

        The file where the AbstractModelStudy is implemented must be executable
        as a script:

        ```
        class MyModelStudy(AbstractModelStudy):
            [...]


        if __name__ == "__main__":
            MyModelStudy.cli_entry_point()
        ```

        Help for the cli can be seen by typing in the terminal:
        ```
        python {FILE}.py --help
        ```
        where FILE is the file where MyModelStudy is defined.
        """
        # self_test()
        self_test_params = []
        self_test_cmd = click.Command("self-test",
                                      callback=cls._cli_self_test,
                                      params=self_test_params)

        # setup_inner_loops()
        setup_inner_loops_params = []
        setup_inner_loops_cmd = click.Command("setup-inner-loops",
                                              callback=cls._cli_setup_inner_loops,
                                              params=setup_inner_loops_params)

        # setup_outer_loop()
        setup_outer_loop_params = []
        setup_outer_loop_cmd = click.Command("setup-outer-loop",
                                             callback=cls._cli_setup_outer_loop,
                                             params=setup_outer_loop_params)

        # init_trial()
        init_trial_params = [click.Option(["-u", "--study-storage-url", "study_storage_url"],
                                          type=str, required=True),
                             click.Option(["-n", "--study-name", "study_name"],
                                          type=str, required=True),
                             click.Option(["-s", "--sampler_path", "sampler_path"],
                                          type=str, required=True),
                             ]
        init_trial_cmd = click.Command("init_trial",
                                       callback=cls._cli_init_trial,
                                       params=init_trial_params)

        # get_hardware_metrics()
        get_hardware_metrics_params = [
            click.Option(["-t", "--trial-path", "trial_path"],
                         type=str, required=True),
            click.Option(["--inner-loop", "inner_loop"],
                         is_flag=True, default=False),
            click.Option(["--outer-loop", "outer_loop"],
                         is_flag=True, default=False),
            click.Option(["-i", "--inner-fold", "inner_fold"],
                         type=int, required=False)]
        get_hardware_metrics_cmd = click.Command("get_hardware_metrics",
                                                 callback=cls._cli_get_hardware_metrics,
                                                 params=get_hardware_metrics_params)

        # get_accuracy_metrics()
        get_accuracy_metrics_params = [
            click.Option(["-t", "--trial-path", "trial_path"],
                         type=str, required=True),
            click.Option(["--inner-loop", "inner_loop"],
                         is_flag=True, default=False),
            click.Option(["--outer-loop", "outer_loop"],
                         is_flag=True, default=False),
            click.Option(["-i", "--inner-fold", "inner_fold"],
                         type=int, required=False)]
        get_accuracy_metrics_cmd = click.Command("get_accuracy_metrics",
                                                 callback=cls._cli_get_accuracy_metrics,
                                                 params=get_accuracy_metrics_params)

        # complete_trial()
        complete_trial_params = [click.Option(
            ["-t", "--trial-path", "trial_path"],
            type=str, required=True),
            click.Option(["--inner-loop", "inner_loop"],
                         is_flag=True, default=False),
            click.Option(["--outer-loop", "outer_loop"],
                         is_flag=True, default=False)
        ]
        complete_trial_cmd = click.Command("complete_trial",
                                           callback=cls._cli_complete_trial,
                                           params=complete_trial_params)

        # Group all commands
        group = click.Group(commands=[self_test_cmd,
                                      setup_inner_loops_cmd,
                                      setup_outer_loop_cmd,
                                      init_trial_cmd,
                                      get_hardware_metrics_cmd,
                                      get_accuracy_metrics_cmd,
                                      complete_trial_cmd])
        group()

    @classmethod
    def _cli_self_test(cls):
        """
        Command-line entry point for the function self_test().
        """
        cls.self_test()

    @classmethod
    def _cli_setup_inner_loops(cls):
        """
        Command-line entry point for the function setup_inner_loops().
        """
        cls.setup_inner_loops()

    @classmethod
    def _cli_setup_outer_loop(cls):
        """
        Command-line entry point for the function setup_outer_loop().
        """
        cls.setup_outer_loop()

    @classmethod
    def _cli_init_trial(cls, study_storage_url: str, study_name: str,
                        sampler_path: str):
        """
        Command-line entry point for the function init_trial().

        Parameters
        ----------
        study_storage_url: str
            URL to study storage.
        study_name: str
            Study name.
        sampler_path: str
            Path to pickled optuna.Sampler.
        """
        sampler = pickle.load(open(sampler_path, "rb"))
        study = optuna.load_study(storage=study_storage_url,
                                  study_name=study_name,
                                  sampler=sampler)
        cls.init_trial(study)

    @classmethod
    def _cli_get_hardware_metrics(cls, trial_path: str, inner_loop: bool,
                                  outer_loop: bool, inner_fold: int):
        """
        Command-line entry point for get_hardware_metrics().

        Parameters
        ----------
        [...]
        """
        trial: optuna.Trial = pickle.load(open(trial_path, "rb"))

        if inner_loop == outer_loop:
            if inner_loop is True:
                raise ValueError("Only one of --inner-loop or --outer-loop "
                                 "should be set.")
            else:
                raise ValueError("One of --inner-loop or --outer-loop should "
                                 "be set.")

        if inner_loop is True:
            loop = "inner"
        else:
            loop = "outer"

        _ = cls.get_hardware_metrics(trial, loop, inner_fold)

    @classmethod
    def _cli_get_accuracy_metrics(cls, trial_path: str, inner_loop: bool,
                                  outer_loop: bool, inner_fold: int):
        """
        Command-line entry point for get_accuracy_metrics().

        Parameters
        ----------
        [...]
        """
        trial: optuna.Trial = pickle.load(open(trial_path, "rb"))

        if inner_loop == outer_loop:
            if inner_loop is True:
                raise ValueError("Only one of --inner-loop or --outer-loop "
                                 "should be set.")
            else:
                raise ValueError("One of --inner-loop or --outer-loop should "
                                 "be set.")

        if inner_loop is True:
            loop = "inner"
        else:
            loop = "outer"

        _ = cls.get_accuracy_metrics(trial, loop, inner_fold)

    @classmethod
    def _cli_complete_trial(cls, trial_path: str, inner_loop: bool,
                            outer_loop: bool):
        """
        Command-line entry point for the function complete_trial().
        """
        trial: optuna.Trial = pickle.load(open(trial_path, "rb"))

        if inner_loop == outer_loop:
            if inner_loop is True:
                raise ValueError("Only one of --inner-loop or --outer-loop "
                                 "should be set.")
            else:
                raise ValueError("One of --inner-loop or --outer-loop should "
                                 "be set.")

        if inner_loop is True:
            loop = "inner"
        else:
            loop = "outer"

        cls.complete_trial(trial, loop)
