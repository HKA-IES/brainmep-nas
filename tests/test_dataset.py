# -*- coding: utf-8 -*-

# import built-in module
import tempfile

# import third-party modules
import pytest
import numpy as np

# import your own module
from brainmepnas.dataset import (Dataset, create_new_dataset,
                                 add_record_to_dataset)


class TestDataset:
    # Parameters for the test dataset
    WINDOW_DURATION = 10    # seconds
    WINDOW_SIZE = 20        # nb of samples
    NB_CHANNELS = 4
    TRAIN_WINDOW_OFFSET = 10
    TEST_WINDOW_OFFSET = 5

    PATIENTS = {"1": {"nb_windows_per_record": 20,
                      "nb_records": 2},
                "2": {"nb_windows_per_record": 40,
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
                               self.WINDOW_DURATION,
                               self.TRAIN_WINDOW_OFFSET,
                               self.TEST_WINDOW_OFFSET,
                               overwrite=True)

            # Generate fake data
            for p_id, p_data in self.PATIENTS.items():
                for _ in range(p_data["nb_records"]):

                    train_x = np.random.rand(p_data["nb_windows_per_record"],
                                             self.WINDOW_SIZE,
                                             self.NB_CHANNELS)
                    train_y = np.random.rand(p_data["nb_windows_per_record"], 1)

                    add_record_to_dataset(temp_dir, p_id, "train",
                                          train_x, train_y)

                    test_x = np.random.rand(p_data["nb_windows_per_record"],
                                             self.WINDOW_SIZE,
                                             self.NB_CHANNELS)
                    test_y = np.random.rand(p_data["nb_windows_per_record"], 1)

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
        assert dataset.nb_patients == len(self.PATIENTS)
        assert dataset.patients == list(self.PATIENTS.keys())
        expected_nb_records_per_patient = {key: val["nb_records"] for key, val in self.PATIENTS.items()}
        assert dataset.nb_records_per_patient == expected_nb_records_per_patient
        expected_total_nb_records = np.sum([val["nb_records"] for val in self.PATIENTS.values()])
        assert dataset.total_nb_records == expected_total_nb_records
        assert dataset.nb_channels == self.NB_CHANNELS
        assert dataset.window_duration == self.WINDOW_DURATION
        assert dataset.train_window_offset == self.TRAIN_WINDOW_OFFSET
        assert dataset.test_window_offset == self.TEST_WINDOW_OFFSET
        assert dataset.data_shape == (self.WINDOW_SIZE, self.NB_CHANNELS)

    def test_get_data_all_patients(self, dataset):
        train_x, train_y = dataset.get_data("all", set="train")

        total_nb_windows = np.sum([val["nb_windows_per_record"] * val["nb_records"]
                                   for val in self.PATIENTS.values()])
        expected_train_x_shape = (total_nb_windows, self.WINDOW_SIZE, self.NB_CHANNELS)
        expected_train_y_shape = (total_nb_windows, 1)

        assert train_x.shape == expected_train_x_shape
        assert train_y.shape == expected_train_y_shape

    def test_get_data_one_patient(self, dataset):
        train_x, train_y = dataset.get_data({"1": "all"}, set="train")

        total_nb_windows = (self.PATIENTS["1"]["nb_windows_per_record"] *
                            self.PATIENTS["1"]["nb_records"])
        expected_train_x_shape = (total_nb_windows, self.WINDOW_SIZE, self.NB_CHANNELS)
        expected_train_y_shape = (total_nb_windows, 1)

        assert train_x.shape == expected_train_x_shape
        assert train_y.shape == expected_train_y_shape

    def test_get_data_all_patients_one_record(self, dataset):
        train_x, train_y = dataset.get_data({"1": [0], "2": [0]},
                                            set="train")
        total_nb_windows = (self.PATIENTS["1"]["nb_windows_per_record"] +
                            self.PATIENTS["2"]["nb_windows_per_record"])
        expected_train_x_shape = (total_nb_windows, self.WINDOW_SIZE, self.NB_CHANNELS)
        expected_train_y_shape = (total_nb_windows, 1)

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

        total_nb_windows = np.sum([val["nb_windows_per_record"] * val["nb_records"]
             for val in self.PATIENTS.values()])
        expected_test_x_shape = (total_nb_windows, self.WINDOW_SIZE, self.NB_CHANNELS)
        expected_test_y_shape = (total_nb_windows, 1)

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

    def test_patients_not_modifiable(self, dataset):
        modified_patients = dataset.patients
        assert modified_patients == dataset.patients

        modified_patients.append("new")

        assert modified_patients != dataset.patients

    def test_nb_records_per_patient_not_modifiable(self, dataset):
        modified_nb_records_per_patient = dataset.nb_records_per_patient
        assert modified_nb_records_per_patient == dataset.nb_records_per_patient

        modified_nb_records_per_patient["new"] = 3

        assert modified_nb_records_per_patient != dataset.nb_records_per_patient
