# -*- coding: utf-8 -*-

# import built-in module
import configparser
import os.path
import shutil
import time

import optuna
# import third-party modules
import pytest

# import your own module
from brainmepnas import AbstractModelStudy
from dummymodelstudy1 import DummyModelStudy1


class TestAbstractModelStudy:
    @staticmethod
    def delete_model_study_directory(model_study: AbstractModelStudy):
        """
        Delete the directory of an AbstractModelStudy implementation, if it
        exists.
        """
        if os.path.isdir(model_study.BASE_DIR):
            shutil.rmtree(model_study.BASE_DIR)

    def test_abstract_class_not_instantiable(self):
        with pytest.raises(RuntimeError):
            AbstractModelStudy()

    def test_abstract_class_method_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._sample_search_space(None)
        with pytest.raises(NotImplementedError):
            AbstractModelStudy.get_accuracy_metrics(None, None)
        with pytest.raises(NotImplementedError):
            AbstractModelStudy.get_hardware_metrics(None, None)

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

    def test_init_trial(self):
        DummyModelStudy1.setup()
        study_storage_url = f"sqlite:///{DummyModelStudy1.BASE_DIR / "study_storage.db"}"
        study_storage = optuna.storages.RDBStorage(study_storage_url)

        study_dir = DummyModelStudy1.BASE_DIR / "outer_fold_0"
        study_name = DummyModelStudy1.NAME + "_outer_fold_0"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        DummyModelStudy1.init_trial(study)

        assert os.path.isdir(study_dir / "trial_0")
        assert os.path.isfile(study_dir / "current_trial.pickle")

        # Properly close connection to the storage
        study_storage.remove_session()
        study_storage.scoped_session.get_bind().dispose()

        self.delete_model_study_directory(DummyModelStudy1)

    def test_save_metrics(self):
        raise NotImplementedError

    def test_complete_trial(self):
        raise NotImplementedError

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
