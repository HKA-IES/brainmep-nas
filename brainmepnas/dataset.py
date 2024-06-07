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

    def __init__(self, directory: Union[str, pathlib.Path]):
        """
        Load an existing dataset.

        Parameters
        ----------
        directory: Union[str, pathlib.Path]
            Path to directory of existing dataset.
        """
        self._directory = directory
        if isinstance(self._directory, str):
            self._directory = pathlib.Path(self._directory)

        if not os.path.isdir(self._directory):
            raise ValueError(f"No dataset found at {self._directory}.")

        # Read properties
        properties_file_path = self._directory / "properties.json"
        with open(properties_file_path, "r") as file:
            properties = json.load(file)

        self._train_window_offset = properties["train_window_offset"]
        self._test_window_offset = properties["test_window_offset"]
        self._window_duration = properties["window_duration"]

        self._train_records = {}
        self._test_records = {}
        self._patients = []
        self._nb_records_per_patient = {}
        for patient, val in properties["patients"].items():
            self._train_records[patient] = val["train_records"]
            self._test_records[patient] = val["test_records"]
            self._patients.append(patient)
            self._nb_records_per_patient[patient] = len(self._train_records[patient])

        sample_data_file_path = (self._directory /
                                 self._train_records[self.patients[0]][0])
        sample = np.load(sample_data_file_path)
        self._data_shape = sample["x"].shape[1:]
        self._window_size = self._data_shape[0]
        self._nb_channels = self._data_shape[1]

    def get_data(self, patients_records: Union[str, Dict[str, Union[str, List[int]]]],
                 set: Literal["train", "test"], shuffle: bool = False,
                 shuffle_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the train or test data and labels for the desired
        patients-records map.

        Parameters
        ----------
        patients_records: Union[str, Dict[str, Union[str, List[int]]]]
            dictionary where each key corresponds to a patient and values are
            a list of records.
            Examples:
                Selecting all records from all patients
                    patients_records = "all"
                Selecting all records from patient "1" and records 0, 1 from patient "2"
                    patients_records = {"1": "all",
                                        "2": [0, 1]}
        set: Literal["train", "test"]
            Get train or test data.
        shuffle: bool
            set to True if data should be shuffled.
        shuffle_seed: int
            random seed for data shuffling.

        Returns
        -------
        x_arr: np.ndarray
            Array of x (input) data.
        y_arr: np.ndarray
            Array of y (labels) data.
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
        for train_recs, test_recs in dataset.split_leave_one_record_out("1"):
            train_x, train_y = dataset.get_data(train_recs, "train")
            test_x, test_y = dataset.get_data(test_recs, "test")

        Parameters
        ----------
        patient: str
            Target patient.

        Yields
        ------
        train_patient_records: dict
            Patients-records map for training.
        test_patient_records: dict
            Patients-records map for testing.
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
        for train_recs, test_recs in dataset.split_leave_one_patient_out():
            train_x, train_y = dataset.get_data(train_recs, "train")
            test_x, test_y = dataset.get_data(test_recs, "test")

        Yields
        ------
        train_patient_records: dict
            Patients-records map for training.
        test_patient_records: dict
            Patients-records map for testing.
        """
        for left_out_patient in self.patients:
            train_dict = {p: "all" for p in self.patients if
                          p != left_out_patient}
            test_dict = {left_out_patient: "all"}
            yield train_dict, test_dict

    @property
    def nb_patients(self) -> int:
        """
        Number of patients in dataset.
        """
        return len(self.patients)

    @property
    def patients(self) -> List[str]:
        """
        List of patients in dataset.
        """
        return self._patients[:]

    @property
    def nb_records_per_patient(self) -> Dict[str, int]:
        """
        Number of records per patient
        """
        return self._nb_records_per_patient.copy()

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
    def window_duration(self) -> float:
        """
        Duration of window, in seconds.
        """
        return self._window_duration

    @property
    def train_window_offset(self) -> float:
        """
        Time offset between the beginning of a window and the beginning of the
        following window for the train set, in seconds.
        """
        return self._train_window_offset

    @property
    def test_window_offset(self) -> float:
        """
        Time offset between the beginning of a window and the beginning of the
        following window for the test set, in seconds.
        """
        return self._test_window_offset

    @property
    def data_shape(self) -> Tuple[int, int]:
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


def create_new_dataset(directory: Union[str, pathlib.Path],
                       window_duration: float,
                       train_window_offset: float,
                       test_window_offset: float,
                       overwrite: bool = False):
    """
    Creates a new dataset where datasets are stored.

    Parameters
    ----------
    directory: Union[str, pathlib.Path]
        location to store the dataset. If directory does not exist, it is
        created. If directory exists, and it contains a dataset, the dataset
        will be overwritten if overwrite=True. Else, an error is returned.
    window_duration: float
        Duration of a window, in seconds.
    train_window_offset: float
        Time offset between the beginning of a window and the beginning of the
        following window for the train set, in seconds. If there is no overlap
        between windows, train_window_offset = window duration.
    test_window_offset: float
        Time offset between the beginning of a window and the beginning of the
        following window for the test set, in seconds.
    overwrite: bool
        Whether an existing dataset with the same name should be overwritten.

    Raises
    ------
    FileExistsError
        If the dataset already exists.
    """
    # Check for validity of arguments
    if not window_duration > 0:
        raise ValueError("window_duration must be > 0 seconds.")

    if not train_window_offset > 0:
        raise ValueError("train_window_offset must be > 0 seconds. For "
                         "non-overlapping samples: "
                         "train_window_offset >= window")

    if not train_window_offset > 0:
        raise ValueError("test_window_offset must be > 0 seconds. For "
                         "non-overlapping samples: "
                         "test_window_offset >= window")

    # Create dataset folder
    if isinstance(directory, str):
        directory = pathlib.Path(directory)

    if os.path.isdir(directory):
        if overwrite:
            warnings.warn(f"A Dataset exists at location {directory}.")
            shutil.rmtree(directory)
            warnings.warn("Overwrite is true, dataset has been overwritten.")
        else:
            raise FileExistsError(f"A Dataset exists at location {directory}."
                                  f"If the dataset is meant to be overwritten,"
                                  f"set overwrite=True.")

    os.mkdir(directory)

    # Save properties to json file
    properties_dict = {"window_duration": window_duration,
                       "train_window_offset": train_window_offset,
                       "test_window_offset": test_window_offset,
                       "patients": {}}

    properties_file_path = directory / "properties.json"
    with open(properties_file_path, "w") as file:
        json.dump(properties_dict, file, indent=4)


def add_record_to_dataset(directory: Union[str, pathlib.Path],
                          patient: str,
                          set: Literal["train", "test"],
                          arr_x: np.ndarray,
                          arr_y: np.ndarray):
    """
    Add new record to dataset.

    Parameters
    ----------
    directory: Union[str, pathlib.Path]
        Path to directory of existing dataset.
    patient: str
        Patient id.
    set: Literal["train", "test"]
        Get train or test data.
    arr_x: np.ndarray
        Array of x (input) data.
    arr_y: np.ndarray
        Array of y (labels) data.
    """
    if isinstance(directory, str):
        directory = pathlib.Path(directory)

    # Check validity of type
    if set not in ["train", "test"]:
        raise ValueError(
            f"Invalid type={set}, should be either test or train.")

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
    new_seizure_id = len(properties["patients"][patient][f"{set}_records"])
    output_file_path = patient_dir / f"{set}_record{new_seizure_id}.npz"
    np.savez(output_file_path, x=arr_x, y=arr_y)

    # Add file path to properties
    output_file_path_as_str = str(output_file_path.relative_to(directory))
    properties["patients"][patient][f"{set}_records"].append(
        output_file_path_as_str)

    # Save updated properties
    with open(properties_file_path, "w") as file:
        json.dump(properties, file, indent=4)
