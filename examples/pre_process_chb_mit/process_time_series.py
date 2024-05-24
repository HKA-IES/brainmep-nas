# -*- coding: utf-8 -*-

# import built-in module
import logging
import time
import pathlib
from typing import Tuple

# import third-party modules
import mne
mne.set_log_level("WARNING")
import numpy as np
import click

# import your own module
from brainmepnas.dataset import create_new_dataset, add_record_to_dataset

# Fixed parameters
WINDOW_DURATION = 4     # seconds
TRAIN_WINDOW_OFFSET = 4    # seconds
TEST_WINDOW_OFFSET = 2     # seconds
CHANNELS = ["F7-T7", "T7-P7", "F8-T8", "T8-P8-0"]

@click.command()
@click.option("-i", "--raw_data_dir",
              type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
              required=True,
              help="Path to raw data directory.")
@click.option("-o", "--output_dir",
              type=click.Path(dir_okay=True, path_type=pathlib.Path),
              required=True,
              help="Path to output directory.")
def process_time_series(raw_data_dir: pathlib.Path, output_dir: pathlib.Path):
    """
    Pre-process CHB-MIT Scalp EEG data and format the output time series as a
    Dataset.

    Parameters
    ----------
    raw_data_dir : pathlib.Path
        Path to raw CHB-MIT Scalp EEG data.
    output_dir : pathlib.Path
        Path to desired output directory.
    """
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Process time series...")
    logging.info(f"raw_data_dir: {raw_data_dir}")
    logging.info(f"output_dir: {output_dir}")

    start_time = time.time()

    records_with_seizures = _get_records_with_seizures(raw_data_dir /
                                                       "RECORDS-WITH-SEIZURES")
    nb_patients = len(records_with_seizures)
    logging.info(f"Found {nb_patients} patients.")

    # Create the dataset with basic infos
    create_new_dataset(directory=output_dir,
                       window_duration=WINDOW_DURATION,
                       train_window_offset=TRAIN_WINDOW_OFFSET,
                       test_window_offset=TEST_WINDOW_OFFSET,
                       overwrite=True)

    # Process each record and add them to the dataset
    for p in records_with_seizures:
        patient_start_time = time.time()

        logging.info(f"Patient {p}/{nb_patients}")
        logging.info(f"Nb of records with at least one seizure: "
                     f"{len(records_with_seizures[p])}")

        # Load data from .edf
        for record_path in records_with_seizures[p]:
            logging.info(f"Processing record {record_path}")

            edf = mne.io.read_raw_edf(raw_data_dir / record_path)
            try:
                edf = edf.pick(CHANNELS)
            except ValueError:
                # Expected channels are not (all) present. Skip this record.
                logging.warning(f"Record {record_path}: all expected channels are not present. Skipping this record.")
                continue

            # Pre-load raw-data to accelerator pre-processing
            edf.load_data()

            annotations = _get_annotations(raw_data_dir / (record_path + ".seizures"))
            edf.set_annotations(annotations)

            # Butterworth filter, order 5, 0.5 Hz to 50 Hz
            iir_params = dict(order=5, ftype="butterworth")
            edf.filter(l_freq=0.5, h_freq=50.0, method="iir",
                       iir_params=iir_params)

            # Training set
            logging.info("\tPreparing training set.")
            train_window_overlap = WINDOW_DURATION - TRAIN_WINDOW_OFFSET
            train_input, train_labels = _get_processed_arrays(edf,
                                      WINDOW_DURATION,
                                      train_window_overlap)
            add_record_to_dataset(output_dir, str(p), "train",
                                  train_input, train_labels)

            # Prepare test set
            logging.info("\tPreparing test set.")
            test_window_overlap = WINDOW_DURATION - TEST_WINDOW_OFFSET
            test_input, test_labels = _get_processed_arrays(edf,
                                                            WINDOW_DURATION,
                                                            test_window_overlap)
            add_record_to_dataset(output_dir, str(p), "test",
                                  test_input, test_labels)

        patient_duration = time.time() - patient_start_time
        logging.info(f"Pre-processed records from patient {p} in {patient_duration} s.")

    duration = time.time() - start_time
    logging.info(f"Completed processing of time series data in {duration} seconds.")


def _get_records_with_seizures(records_with_seizures_file_path: pathlib.Path) -> dict:
    """
    Parse RECORD-WITH-SEIZURES file from the CHB-MIT Scalp EEG data into a
    dictionary containing a list of paths of files with seizures for each
    patient.

    Parameters
    ----------
    records_with_seizures_file_path : pathlib.Path
        Path to RECORD-WITH-SEIZURES file from the CHB-MIT Scalp EEG data.

    Returns
    -------
    records_with_seizures : dict
        Keys are patient ids and values are lists of paths of files with
        seizures.
    """
    records_with_seizures = dict()
    with open(records_with_seizures_file_path, "r") as f:
        for line in f:
            if line == "\n":
                continue

            # Remove \n
            line = line[:-1]

            try:
                records_with_seizures[int(line[3:5])].append(line)
            except KeyError:
                records_with_seizures[int(line[3:5])] = [line]
    return records_with_seizures

def _get_annotations(annotations_path: pathlib.Path) -> mne.Annotations:
    """
    Process annotations file into mne.Annotations format.

    Parameters
    ----------
    annotations_path : pathlib.Path
        Path to chbxx_yy.edf.seizures file.

    Returns
    -------
    annotations : mne.Annotations
        Processed annotations.
    """
    with open(annotations_path, "rb") as f:
        data = f.read()

        i = 37
        # Adding a dummy zero element to allow calculation of
        # start_times to be equivalent for the first seizure as well
        # as the subsequent seizures.
        start_times = [0]
        durations = [0]
        while data[i] == 0xEC:
            time_since_previous_end = data[i + 1] * 256 + data[i + 4]
            start_times.append(start_times[-1] + durations[-1]
                               + time_since_previous_end)
            new_length = data[i + 12]
            if new_length == 255:
                logging.warning("Length of 255 seconds found, might "
                                "indicate an overflow.")
            durations.append(new_length)
            i += 16
    return mne.Annotations(start_times[1:], durations[1:], "ictal")

def _get_processed_arrays(edf,
                          window_length: float,
                          window_overlap: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process raw data in windows with specified length and overlap, and apply
    scaling.

    Parameters
    ----------
    edf : mne.io.edf.edf.RawEDF
        Object for .edf file.
    window_length : float
        Window length in seconds.
    window_overlap : float
        Window overlap in seconds.

    Returns
    -------
    x: np.ndarray
        Array of windowed input data, channels last format.
    y: np.ndarray
        Array of windowed labels, with 0 for interictal and 1 for ictal.
    """
    epochs = mne.make_fixed_length_epochs(edf,
                                                duration=window_length,
                                                overlap=window_overlap)
    x = np.array(epochs.get_data(), dtype=np.float16)

    scaler = mne.decoding.Scaler(scalings="mean")
    x_scaled = scaler.fit_transform(x)

    # we want channels last
    x_scaled = np.swapaxes(x_scaled, 1, 2)

    y = np.zeros((x_scaled.shape[0], 1),
                            dtype=np.float16)

    nb_epochs_ictal = 0
    for i, annotations in enumerate(epochs.get_annotations_per_epoch()):
        if len(annotations) > 0:
            logging.debug(f"\tEpoch {i} is ictal.")
            nb_epochs_ictal += 1
            y[i] = 1

    logging.info(f"\t{nb_epochs_ictal}/{len(epochs)} epochs "
                 f"are ictal")

    return x, y

if __name__ == '__main__':
    process_time_series()
