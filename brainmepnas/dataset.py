# -*- coding: utf-8 -*-

# import built-in module
import json
import pathlib
import os
import shutil
from typing import Iterator, Tuple, List, Union, Dict, Literal
import warnings

# import third-party modules
import numpy as np
import sklearn.utils

# import your own module


class Dataset:
    """
    Represents a Dataset which can be used for training and testing of any
    model.

    A dataset is stored in a specific directory.

    To load a dataset: dataset = Dataset(directory)
    To create a dataset: create_new_dataset(directory, ...)
    To add a record to an existing dataset: add_record_to_dataset(directory, ...)

    Dataset creation and modification is separated from the Dataset class
    because modifying a data set should be done in different contexts than
    loading a dataset.
    """

    # TODO: Bug, "patients" can be modified

    def __init__(self, directory: Union[str, pathlib.Path]):
        self._directory = directory
        if isinstance(self._directory, str):
            self._directory = pathlib.Path(self._directory)

        if not os.path.isdir(self._directory):
            raise ValueError(
                f"No dataset found at location {self._directory}.")

        # Read properties
        properties_file_path = self._directory / "properties.json"
        with open(properties_file_path, "r") as file:
            properties = json.load(file)

        self._sampling_frequency = properties["sampling_frequency"]
        self._train_window_overlap = properties["train_window_overlap"]
        self._test_window_overlap = properties["test_window_overlap"]

        self._train_records = {}
        self._test_records = {}
        self._patients = []
        self._nb_records_per_patient = {}
        for id, val in properties["patients"].items():
            self._train_records[id] = val["train_records"]
            self._test_records[id] = val["test_records"]
            self._patients.append(id)
            self._nb_records_per_patient[id] = len(self._train_records[id])

        sample_data_file_path = (self._directory /
                                 self._train_records[self.patients[0]][0])
        sample = np.load(sample_data_file_path)
        self._data_shape = sample["x"].shape[1:]
        self._window_length = self._data_shape[0]
        self._nb_channels = self._data_shape[1]

    def get_data(self, patients_records: Union[
        str, Dict[str, Union[str, List[int]]]],
                 set: Literal["train", "test"], shuffle: bool = False,
                 shuffle_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the train or test data and labels for the desired patients-records map.

        :param patients_records: dictionary where each key corresponds to a patient
        and values are a list of records.
            Examples:
                Selecting all records from all patients
                    patients_records = "all"
                Selecting all records from patient "1" and records 0, 1 from patient "2"
                    patients_records = {"1": "all",
                                        "2": [0, 1]}
        :param set: "test" or "train"
        :param shuffle: set to True if data should be shuffled.
        :param shuffle_seed: random seed for data shuffling.
        :return train_x, train_y
        """
        if patients_records == "all":
            patients_records = {p: "all" for p in self.patients}

        if set == "train":
            records_list = self._train_records
        elif set == "test":
            records_list = self._test_records
        else:
            raise ValueError(f"Unsupported set={set}. "
                             f"Should be either train or test.")

        x_list = list()
        y_list = list()

        for patient, records in patients_records.items():
            if patient not in self.patients:
                raise ValueError(f"Patient {patient} is not valid. "
                                 f"Valid values are {self.patients}.")
            if records == "all":
                records = [r for r in
                           range(self.nb_records_per_patient[patient])]

            for r in records:
                try:
                    record_path = (self._directory /
                                   pathlib.Path(records_list[patient][r]))
                except IndexError:
                    raise ValueError(f"Record {r} is not valid. Valid values "
                                     f"for patient {patient} are between 0 and "
                                     f"{self.nb_records_per_patient[patient]}")
                single_record = np.load(record_path)
                x_list.append(single_record["x"])
                y_list.append(single_record["y"])

        x_arr = np.concatenate(x_list)
        y_arr = np.concatenate(y_list)

        if shuffle:
            x_arr, y_arr = sklearn.utils.shuffle(x_arr, y_arr,
                                                 random_state=shuffle_seed)

        return x_arr, y_arr

    def split_leave_one_record_out(self, patient: str) \
            -> Iterator[Tuple[Dict, Dict]]:
        """
        Generates a train/test split to execute leave-one(-record)-out
        cross-validation for a single patient.

        Example use:
        for train, test in dataset.split_leave_one_record_out("1"):
            train_x, train_y = dataset.get_data(train)
            test_x, test_y = dataset.get_data(test)

        :param patient: target patient.
        :return: train_patients_records, test_patients_records
        """
        if patient not in self.patients:
            raise ValueError(f"Patient {patient} is not valid. "
                             f"Valid values are {self.patients}.")

        nb_records = self.nb_records_per_patient[patient]

        for left_out_record in range(nb_records):
            train_records = [r for r in range(nb_records) if
                             r != left_out_record]
            test_records = [left_out_record]
            yield {patient: train_records}, {patient: test_records}

    def split_leave_one_patient_out(self) -> Iterator[Tuple[Dict, Dict]]:
        """
        Generates a train/test split to execute leave-one(-patient)-out
        cross-validation for.

        Example use:
        for train, test in dataset.split_leave_one_patient_out():
            train_x, train_y = dataset.get_data(train)
            test_x, test_y = dataset.get_data(test)

        :return: train_patients_records, test_patients_records
        """
        for left_out_patient in self.patients:
            train_dict = {p: "all" for p in self.patients if
                          p != left_out_patient}
            test_dict = {left_out_patient: "all"}
            yield train_dict, test_dict

    @property
    def sampling_frequency(self) -> float:
        """
        Sampling frequency of the data in Hz.
        """
        return self._sampling_frequency

    @property
    def nb_patients(self) -> int:
        """
        Number of patients in dataset.
        """
        return len(self.patients)

    @property
    def patients(self) -> list[str]:
        """
        List of patients in dataset.
        """
        return self._patients

    @property
    def nb_records_per_patient(self) -> dict[str: int]:
        """
        Number of records per patient
        """
        return self._nb_records_per_patient

    @property
    def total_nb_records(self) -> int:
        """
        Total number of records.
        """
        return sum(self.nb_records_per_patient.values())

    @property
    def nb_channels(self) -> int:
        """
        Number of EEG channels.
        """
        return self._nb_channels

    @property
    def window_length(self) -> int:
        """
        Length in samples of the window.
        """
        return self._window_length

    @property
    def train_window_overlap(self) -> int:
        """
        Length in samples of the window overlap for the train set.
        """
        return self._train_window_overlap

    @property
    def test_window_overlap(self) -> int:
        """
        Length in samples of the window overlap for the test set.
        """
        return self._test_window_overlap

    @property
    def data_shape(self) -> tuple[int, int]:
        """
        Shape of each data point (samples, channels)
        """
        return self._data_shape

    @property
    def directory(self) -> pathlib.Path:
        """
        Directory of the dataset.
        """
        return self._directory


def create_new_dataset(directory: Union[str, pathlib.Path], sampling_frequency: int,
                       train_window_overlap: int, test_window_overlap: int,
                       overwrite: bool = False):
    """
    Creates a new dataset where datasets are stored.

    :param directory: location to store the dataset. If directory does not
    exist, it is created. If directory exists, and it contains a dataset, the
    dataset will be overwritten if overwrite=True. Else, an error is returned.
    :param overwrite: whether an existing dataset with the same name should be
    overwritten.
    :param sampling_frequency:
    :param train_window_overlap:
    :param test_window_overlap:
    """
    # Create dataset folder
    if isinstance(directory, str):
        directory = pathlib.Path(directory)

    if os.path.isdir(directory):
        warnings.warn(f"A Dataset exists at location {directory}.")
        if overwrite:
            shutil.rmtree(directory)
            warnings.warn("Overwrite is true, dataset has been overwritten.")
        else:
            warnings.warn("Overwrite is False, dataset creation aborted.")
            return None

    os.mkdir(directory)

    # Save properties to json file
    properties_dict = {"sampling_frequency": sampling_frequency,
                       "train_window_overlap": train_window_overlap,
                       "test_window_overlap": test_window_overlap,
                       "patients": {}}

    properties_file_path = directory / "properties.json"
    with open(properties_file_path, "w") as file:
        json.dump(properties_dict, file, indent=4)


def add_record_to_dataset(directory: Union[str, pathlib.Path], patient: str, type: str,
                          arr_x: np.ndarray, arr_y: np.ndarray):
    """
    Add new record to dataset.

    :param directory: dataset directory.
    :param patient: Patient id.
    :param type: train or test.
    :param arr_x:
    :param arr_y:
    """
    if isinstance(directory, str):
        directory = pathlib.Path(directory)

    # Check validity of type
    if type not in ["train", "test"]:
        raise ValueError(
            f"Invalid type={type}, should be either test or train.")

    # Load Dataset properties
    properties_file_path = directory / "properties.json"
    with open(properties_file_path, "r") as file:
        properties = json.load(file)

    # If no records exist for this patient, create a new folder
    patient_dir = directory / f"patient{patient}"
    if not os.path.isdir(patient_dir):
        properties["patients"][patient] = {"train_records": [],
                                           "test_records": []}
        os.mkdir(patient_dir)

    # Save data
    new_seizure_id = len(properties["patients"][patient][f"{type}_records"])
    output_file_path = patient_dir / f"{type}_record{new_seizure_id}.npz"
    np.savez(output_file_path, x=arr_x, y=arr_y)

    # Add file path to properties
    output_file_path_as_str = str(output_file_path.relative_to(directory))
    properties["patients"][patient][f"{type}_records"].append(
        output_file_path_as_str)

    # Save updated properties
    with open(properties_file_path, "w") as file:
        json.dump(properties, file, indent=4)