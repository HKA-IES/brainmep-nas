# -*- coding: utf-8 -*-

# import built-in module
import os.path
import shutil
import pickle
from typing import Type

# import third-party modules
import pytest
import optuna

# import your own module
from brainmepnas import AbstractModelStudy
from dummymodelstudy1 import DummyModelStudy1


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
        self.delete_model_study_directory(DummyModelStudy1)

    def test_abstract_class_not_instantiable(self):
        with pytest.raises(RuntimeError):
            AbstractModelStudy()

    def test_abstract_class_method_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._sample_search_space(None)
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._get_accuracy_metrics(None, None)
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._get_hardware_metrics(None, None)

    def test_implementation_not_instantiable(self):
        with pytest.raises(RuntimeError):
            DummyModelStudy1()

    def test_setup_folder_already_exists(self):
        DummyModelStudy1.setup()

        with pytest.raises(FileExistsError):
            DummyModelStudy1.setup()

        self.delete_model_study_directory(DummyModelStudy1)

    def test_setup_all_files_and_dirs_created(self):
        DummyModelStudy1.setup()
        base_dir = DummyModelStudy1.BASE_DIR

        assert os.path.isdir(base_dir)
        assert os.path.isfile(base_dir / "study_storage.db")
        assert os.path.isfile(base_dir / "run_model_study.sh")
        for outer_fold in range(DummyModelStudy1.N_FOLDS):
            outer_fold_dir = DummyModelStudy1.BASE_DIR / f"outer_fold_{outer_fold}"
            assert os.path.isdir(outer_fold_dir)
            assert os.path.isfile(outer_fold_dir / "sampler.pickle")
            assert os.path.isfile(outer_fold_dir / "run_trial.sh")
            assert os.path.isfile(outer_fold_dir / "run_study.sh")

        self.delete_model_study_directory(DummyModelStudy1)

    def test_init_trial(self, mocker):
        DummyModelStudy1.setup()
        study_storage_url = f"sqlite:///{DummyModelStudy1.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_dir = DummyModelStudy1.BASE_DIR / "outer_fold_0"
        study_name = DummyModelStudy1.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        mocked_method = mocker.patch(
            "dummymodelstudy1.DummyModelStudy1._sample_search_space")
        mocked_method.return_value = None

        trial = DummyModelStudy1.init_trial(study)

        mocked_method.assert_called_once()
        assert os.path.isdir(study_dir / "trial_0")
        assert os.path.isfile(study_dir / "current_trial.pickle")
        assert isinstance(trial, optuna.Trial)
        # assert isinstance(pickle.load(open(study_dir / "current_trial.pickle", "rb")), optuna.Trial)

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(DummyModelStudy1)

    def test_get_accuracy_metrics(self, mocker):
        DummyModelStudy1.setup()
        study_storage_url = f"sqlite:///{DummyModelStudy1.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = DummyModelStudy1.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = DummyModelStudy1.init_trial(study)
        trial_dir = DummyModelStudy1.get_trial_dir(trial)

        mocked_method = mocker.patch("dummymodelstudy1.DummyModelStudy1._get_accuracy_metrics")
        mocked_method.return_value = None

        am = DummyModelStudy1.get_accuracy_metrics(trial, 1)

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_1_accuracy_metrics.pickle", "rb")) == am

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(DummyModelStudy1)

    def test_get_hardware_metrics(self, mocker):
        DummyModelStudy1.setup()
        study_storage_url = f"sqlite:///{DummyModelStudy1.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = DummyModelStudy1.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = DummyModelStudy1.init_trial(study)
        trial_dir = DummyModelStudy1.get_trial_dir(trial)

        mocked_method = mocker.patch("dummymodelstudy1.DummyModelStudy1._get_hardware_metrics")
        mocked_method.return_value = None

        hm = DummyModelStudy1.get_hardware_metrics(trial, 1)

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_1_hardware_metrics.pickle", "rb")) == hm

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(DummyModelStudy1)

    def test_get_combined_metrics(self, mocker):
        DummyModelStudy1.setup()
        study_storage_url = f"sqlite:///{DummyModelStudy1.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = DummyModelStudy1.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = DummyModelStudy1.init_trial(study)
        trial_dir = DummyModelStudy1.get_trial_dir(trial)

        mocked_method = mocker.patch("dummymodelstudy1.DummyModelStudy1._get_combined_metrics")
        mocked_method.return_value = None

        cm = DummyModelStudy1.get_combined_metrics(trial, 1, None, None)

        mocked_method.assert_called_once()
        assert pickle.load(open(trial_dir / "inner_fold_1_combined_metrics.pickle", "rb")) == cm

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(DummyModelStudy1)

    def test_complete_trial(self):
        DummyModelStudy1.setup()
        study_storage_url = f"sqlite:///{DummyModelStudy1.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = DummyModelStudy1.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = DummyModelStudy1.init_trial(study)
        trial_dir = DummyModelStudy1.get_trial_dir(trial)

        DummyModelStudy1.get_hardware_metrics(trial)
        DummyModelStudy1.get_accuracy_metrics(trial, 0)
        DummyModelStudy1.get_accuracy_metrics(trial, 1)

        assert study.trials[-1].state == optuna.trial.TrialState.RUNNING
        DummyModelStudy1.complete_trial(study, trial)
        assert study.trials[-1].state == optuna.trial.TrialState.COMPLETE

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(DummyModelStudy1)

    def test_complete_trial_metrics_missing(self):
        DummyModelStudy1.setup()
        study_storage_url = f"sqlite:///{DummyModelStudy1.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_name = DummyModelStudy1.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        trial = DummyModelStudy1.init_trial(study)
        trial_dir = DummyModelStudy1.get_trial_dir(trial)

        #DummyModelStudy1.get_hardware_metrics(trial)
        #DummyModelStudy1.get_accuracy_metrics(trial, 0)
        #DummyModelStudy1.get_accuracy_metrics(trial, 1)

        assert study.trials[-1].state == optuna.trial.TrialState.RUNNING
        DummyModelStudy1.complete_trial(study, trial)
        assert study.trials[-1].state == optuna.trial.TrialState.FAIL

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(DummyModelStudy1)

    def test_get_outer_fold(self):
        raise NotImplementedError

    def test_get_trial_dir(self):
        raise NotImplementedError

    def test_cli(self):
        raise NotImplementedError

    def test_cli_setup(self):
        raise NotImplementedError

    def test_cli_init_trial(self):
        raise NotImplementedError

    def test_cli_get_hardware_metrics(self):
        raise NotImplementedError

    def test_cli_get_accuracy_metrics(self):
        raise NotImplementedError

    def test_cli_complete_trial(self):
        raise NotImplementedError
