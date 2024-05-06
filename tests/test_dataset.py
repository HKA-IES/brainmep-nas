# -*- coding: utf-8 -*-

# import built-in module
import pathlib
import tempfile

# import third-party modules
import pytest
import numpy as np

# import your own module
from brainmepnas.dataset import (Dataset, create_new_dataset,
                                 add_record_to_dataset)


class TestDataset:
    # Parameters for the test dataset
    SAMPLING_FREQUENCY = 10
    WINDOW_LENGTH = 10
    NB_CHANNELS = 4
    TRAIN_WINDOW_OVERLAP = 0
    TEST_WINDOW_OVERLAP = 5

    PATIENTS = {"1": {"record_length": 20 * SAMPLING_FREQUENCY,
                      "nb_records": 2},
                "2": {"record_length": 40 * SAMPLING_FREQUENCY,
                      "nb_records": 4}}

    @pytest.fixture(scope="class", autouse=True)
    def dataset(self):
        """
        Automatic generation of test dataset.

        This indirectly tests the create_new_dataset and add_record_to_dataset
        methods of brainmepnas.dataset.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            create_new_dataset(temp_dir,
                               self.SAMPLING_FREQUENCY,
                               self.TRAIN_WINDOW_OVERLAP,
                               self.TEST_WINDOW_OVERLAP,
                               overwrite=True)

            # Generate fake data
            for p_id, p_data in self.PATIENTS.items():
                for _ in range(p_data["nb_records"]):
                    train_nb_windows = int(
                        (p_data["record_length"] - self.WINDOW_LENGTH) /
                        (self.WINDOW_LENGTH - self.TRAIN_WINDOW_OVERLAP) + 1)

                    train_x = np.random.rand(train_nb_windows,
                                             self.WINDOW_LENGTH,
                                             self.NB_CHANNELS)
                    train_y = np.random.rand(train_nb_windows, 1)

                    add_record_to_dataset(temp_dir, p_id, "train",
                                          train_x, train_y)

                    test_nb_windows = int(
                        (p_data["record_length"] - self.WINDOW_LENGTH) /
                        (self.WINDOW_LENGTH - self.TEST_WINDOW_OVERLAP) + 1)

                    test_x = np.random.rand(test_nb_windows,
                                            self.WINDOW_LENGTH,
                                            self.NB_CHANNELS)
                    test_y = np.random.rand(test_nb_windows, 1)

                    add_record_to_dataset(temp_dir, p_id, "test",
                                          test_x, test_y)
            # yielding Dataset() allows all test functions to access the
            # Dataset() instance in their parameters.
            # When all tests are run, the code goes past the yield instruction
            # and exits the context manager, which destroys temp_dir.
            yield Dataset(temp_dir)

    def test_init_bad_dir(self):
        """
        When Dataset is initialized with a bad directory,
        - ValueError is raised.
        """
        with pytest.raises(ValueError):
            _ = Dataset("bad_dir")

    def test_init(self, dataset):
        """
        When Dataset is initialized with a valid name,
        - all attributes correspond to the data attributes.
        """
        assert dataset.sampling_frequency == 10
        assert dataset.nb_patients == 2
        assert dataset.patients == ["1", "2"]
        assert dataset.nb_records_per_patient == {"1": 2, "2": 4}
        assert dataset.total_nb_records == 6
        assert dataset.nb_channels == 4
        assert dataset.window_length == 10
        assert dataset.train_window_overlap == 0
        assert dataset.test_window_overlap == 5
        assert dataset.data_shape == (10, 4)

    def test_get_data_all_patients(self, dataset):
        train_x, train_y = dataset.get_data("all", set="train")

        # 200/10 * 2 = 40 samples for patient 1
        # 400/10 * 4 = 160 samples for patient 2
        expected_train_x_shape = (40+160, 10, 4)
        expected_train_y_shape = (40+160, 1)

        assert train_x.shape == expected_train_x_shape
        assert train_y.shape == expected_train_y_shape

    def test_get_data_one_patient(self, dataset):
        train_x, train_y = dataset.get_data({"1": "all"}, set="train")

        # 200/10 * 2 = 40 samples for patient 1
        expected_train_x_shape = (40, 10, 4)
        expected_train_y_shape = (40, 1)

        assert train_x.shape == expected_train_x_shape
        assert train_y.shape == expected_train_y_shape

    def test_get_data_all_patients_one_record(self, dataset):
        train_x, train_y = dataset.get_data({"1": [0], "2": [0]},
                                            set="train")
        # 200/10 * 1 = 20 samples for patient 1
        # 400/10 * 1 = 40 samples for patient 2
        expected_train_x_shape = (20+40, 10, 4)
        expected_train_y_shape = (20+40, 1)

        assert train_x.shape == expected_train_x_shape
        assert train_y.shape == expected_train_y_shape

    def test_get_data_bad_patient(self, dataset):
        with pytest.raises(ValueError):
            _, _ = dataset.get_data({"450": "all"}, set="train")

    def test_get_data_bad_record(self, dataset):
        with pytest.raises(ValueError):
            _, _ = dataset.get_data({"1": [450]}, set="train")

    def test_get_data_shuffle(self, dataset):
        train_x, train_y = dataset.get_data({"1": "all"}, set="train")
        shuffled_train_x, shuffled_train_y = dataset.get_data({"1": "all"},
                                                              set="train",
                                                              shuffle=True)

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(shuffled_train_x, train_x)
            np.testing.assert_array_equal(shuffled_train_y, train_y)

    def test_get_data_test(self, dataset):
        test_x, test_y = dataset.get_data("all", set="test")

        # patient 1: 200 total samples, 5 samples overlap, 10 window length,
        # 39 windows per seizure => 39*2 elements
        # patient 2: 400 total samples, 5 samples overlap, 10 window length,
        # 79 windows per seizure => 79*4 elements

        expected_test_x_shape = (39*2+79*4, 10, 4)
        expected_test_y_shape = (39*2+79*4, 1)

        assert test_x.shape == expected_test_x_shape
        assert test_y.shape == expected_test_y_shape

    def test_split_leave_one_record_out(self, dataset):
        split = dataset.split_leave_one_record_out("2")

        iter0 = next(split)
        assert iter0[0] == {"2": [1, 2, 3]}
        assert iter0[1] == {"2": [0]}

        iter1 = next(split)
        assert iter1[0] == {"2": [0, 2, 3]}
        assert iter1[1] == {"2": [1]}

        iter2 = next(split)
        assert iter2[0] == {"2": [0, 1, 3]}
        assert iter2[1] == {"2": [2]}

        iter3 = next(split)
        assert iter3[0] == {"2": [0, 1, 2]}
        assert iter3[1] == {"2": [3]}

        with pytest.raises(StopIteration):
            next(split)

    def test_split_leave_one_record_out_bad_patient(self, dataset):
        with pytest.raises(ValueError):
            split = dataset.split_leave_one_record_out("bad")
            next(split)

    def test_split_leave_one_patient_out(self, dataset):
        split = dataset.split_leave_one_patient_out()

        iter0 = next(split)
        assert iter0[0] == {"2": "all"}
        assert iter0[1] == {"1": "all"}

        iter1 = next(split)
        assert iter1[0] == {"1": "all"}
        assert iter1[1] == {"2": "all"}

        with pytest.raises(StopIteration):
            next(split)
