# -*- coding: utf-8 -*-

# import built-in module
import os.path
import shutil
import pickle
from typing import Type

# import third-party modules
import pytest
import optuna
from click.testing import CliRunner

# import your own module
from brainmepnas import AbstractModelStudy
from goodmodelstudy import GoodModelStudy
from badmodelstudy import BadModelStudy


class TestAbstractModelStudy:

    @staticmethod
    def delete_model_study_directory(model_study: Type[AbstractModelStudy]):
        """
        Delete the directory of an AbstractModelStudy implementation, if it
        exists.
        """
        if os.path.isdir(model_study.BASE_DIR):
            shutil.rmtree(model_study.BASE_DIR)

    @pytest.fixture(scope="session", autouse=True)
    def run_before_all_tests(self):
        self.delete_model_study_directory(GoodModelStudy)

    def test_abstract_class_not_instantiable(self):
        with pytest.raises(RuntimeError):
            AbstractModelStudy()

    def test_abstract_class_method_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._sample_search_space(None)
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._get_accuracy_metrics(None, None,
                                                     None, None)
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._get_hardware_metrics(None, None,
                                                     None, None)

    def test_implementation_not_instantiable(self):
        with pytest.raises(RuntimeError):
            GoodModelStudy()

    def test_self_test_pass(self):
        GoodModelStudy.self_test()

    def test_self_test_fail(self):
        with pytest.raises(Exception):
            BadModelStudy.self_test()

    def test_setup_inner_loops_folder_already_exists(self):
        GoodModelStudy.setup_inner_loops()

        with pytest.raises(FileExistsError):
            GoodModelStudy.setup_inner_loops()

        self.delete_model_study_directory(GoodModelStudy)

    def test_setup_inner_loops_all_files_and_dirs_created(self):
        GoodModelStudy.setup_inner_loops()
        base_dir = GoodModelStudy.BASE_DIR

        assert os.path.isdir(base_dir)
        assert os.path.isfile(base_dir / "study_storage.db")
        assert os.path.isfile(base_dir / "run_all_inner_loops.sh")
        for outer_fold in range(GoodModelStudy.N_FOLDS):
            outer_fold_dir = GoodModelStudy.BASE_DIR / f"outer_fold_{outer_fold}"
            assert os.path.isdir(outer_fold_dir)
            assert os.path.isfile(outer_fold_dir / "sampler.pickle")
            assert os.path.isfile(outer_fold_dir / "run_trial.sh")
            assert os.path.isfile(outer_fold_dir / "run_inner_loop.sh")

        self.delete_model_study_directory(GoodModelStudy)

    def test_setup_outer_loop_files_already_exist(self):
        GoodModelStudy.setup_inner_loops()
        GoodModelStudy.setup_outer_loop()

        # No exception is raised
        GoodModelStudy.setup_outer_loop()

        self.delete_model_study_directory(GoodModelStudy)

    def test_setup_outer_loop_all_files_created(self):
        GoodModelStudy.setup_inner_loops()
        GoodModelStudy.setup_outer_loop()
        base_dir = GoodModelStudy.BASE_DIR

        assert os.path.isfile(base_dir / "run_outer_loop.sh")
        for outer_fold in range(GoodModelStudy.N_FOLDS):
            assert os.path.isfile(base_dir / f"outer_fold_{outer_fold}" / "process_pareto_set.sh")

        self.delete_model_study_directory(GoodModelStudy)

    def test_init_trial(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_dir = GoodModelStudy.BASE_DIR / "outer_fold_0"
        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        mocked_method = mocker.patch(
            "goodmodelstudy.GoodModelStudy._sample_search_space")
        mocked_method.return_value = None

        trial = GoodModelStudy.init_trial(study)

        mocked_method.assert_called_once()
        assert os.path.isdir(study_dir / "trial_0")
        assert os.path.isfile(study_dir / "current_trial.pickle")
        assert isinstance(trial, optuna.Trial)
        # assert isinstance(pickle.load(open(study_dir / "current_trial.pickle", "rb")), optuna.Trial)

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_accuracy_metrics_inner_loop(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_accuracy_metrics")
        mocked_method.return_value = None

        am = GoodModelStudy.get_accuracy_metrics(trial, "inner", 1)

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_1_accuracy_metrics.pickle", "rb")) == am

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_accuracy_metrics_inner_loop_all_folds(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_accuracy_metrics")
        mocked_method.return_value = None

        am = GoodModelStudy.get_accuracy_metrics(trial, "inner", None)

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_all_accuracy_metrics.pickle", "rb")) == am

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_accuracy_metrics_outer_loop(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_accuracy_metrics")
        mocked_method.return_value = None

        am = GoodModelStudy.get_accuracy_metrics(trial, "outer")

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "outer_fold_0_accuracy_metrics.pickle", "rb")) == am

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_hardware_metrics_inner_loop(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_hardware_metrics")
        mocked_method.return_value = None

        hm = GoodModelStudy.get_hardware_metrics(trial, "inner", 1)

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_1_hardware_metrics.pickle", "rb")) == hm

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_hardware_metrics_inner_loop_all_folds(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_hardware_metrics")
        mocked_method.return_value = None

        hm = GoodModelStudy.get_hardware_metrics(trial, "inner")

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_all_hardware_metrics.pickle", "rb")) == hm

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_hardware_metrics_outer_loop(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_hardware_metrics")
        mocked_method.return_value = None

        hm = GoodModelStudy.get_hardware_metrics(trial, "outer")

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "outer_fold_0_hardware_metrics.pickle", "rb")) == hm

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_combined_metrics_inner_loop(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_combined_metrics")
        mocked_method.return_value = None

        cm = GoodModelStudy.get_combined_metrics(None, None,
                                                 trial, "inner", 1)

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_1_combined_metrics.pickle", "rb")) == cm

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_combined_metrics_inner_loop_all_folds(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_combined_metrics")
        mocked_method.return_value = None

        cm = GoodModelStudy.get_combined_metrics(None, None,
                                                 trial, "inner")

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_all_combined_metrics.pickle", "rb")) == cm

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_get_combined_metrics_outer_loop(self, mocker):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        mocked_method = mocker.patch("goodmodelstudy.GoodModelStudy._get_combined_metrics")
        mocked_method.return_value = None

        cm = GoodModelStudy.get_combined_metrics(None, None,
                                                 trial, "outer")

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "outer_fold_0_combined_metrics.pickle", "rb")) == cm

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_complete_trial_inner_loop(self):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)

        GoodModelStudy.get_hardware_metrics(trial, "inner")
        GoodModelStudy.get_accuracy_metrics(trial, "inner", 1)
        GoodModelStudy.get_accuracy_metrics(trial, "inner", 2)

        assert study.trials[-1].state == optuna.trial.TrialState.RUNNING
        GoodModelStudy.complete_trial(trial, "inner")
        assert study.trials[-1].state == optuna.trial.TrialState.COMPLETE
        assert os.path.isfile(GoodModelStudy.BASE_DIR / "outer_fold_0" 
                              / "inner_loop_metrics.csv")

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_complete_trial_inner_loop_metrics_missing(self):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)

        #DummyModelStudy1.get_hardware_metrics(trial)
        #DummyModelStudy1.get_accuracy_metrics(trial, 0)
        #DummyModelStudy1.get_accuracy_metrics(trial, 1)

        assert study.trials[-1].state == optuna.trial.TrialState.RUNNING
        GoodModelStudy.complete_trial(trial, "inner")
        assert study.trials[-1].state == optuna.trial.TrialState.FAIL

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)

    def test_complete_trial_outer_loop(self):
        GoodModelStudy.setup_inner_loops()
        study_storage_url = f"sqlite:///{GoodModelStudy.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = GoodModelStudy.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = GoodModelStudy.init_trial(study)
        trial_dir = GoodModelStudy.BASE_DIR / "outer_fold_0" / f"trial_{trial.number}"

        GoodModelStudy.get_hardware_metrics(trial, "inner")
        GoodModelStudy.get_accuracy_metrics(trial, "outer")
        GoodModelStudy.complete_trial(trial, "outer")

        assert os.path.isfile(GoodModelStudy.BASE_DIR / "outer_loop_metrics.csv")

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(GoodModelStudy)
