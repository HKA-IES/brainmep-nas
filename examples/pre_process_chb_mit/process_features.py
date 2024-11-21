# -*- coding: utf-8 -*-

# import built-in module
import logging
import time
import pathlib

# import third-party modules
import mne
mne.set_log_level("WARNING")
import numpy as np
import click
import scipy

# import your own module
from brainmepnas import Dataset
from brainmepnas.dataset import create_new_dataset, add_record_to_dataset

# Fixed parameters
SAMPLING_FREQUENCY = 256


@click.command()
@click.option("-i", "--time_series_dataset_dir",
              type=click.Path(exists=True, dir_okay=True,
                              path_type=pathlib.Path),
              required=True,
              help="Path to directory of time series dataset "
                   "(created with process_time_series.py).")
@click.option("-o", "--output_dir",
              type=click.Path(dir_okay=True, path_type=pathlib.Path),
              required=True,
              help="Path to output directory.")
def process_features(time_series_dataset_dir: pathlib.Path,
                     output_dir: pathlib.Path):
    """
    Pre-process CHB-MIT Scalp EEG data and format the output time series as a
    Dataset.

    Parameters
    ----------
    time_series_dataset_dir : pathlib.Path
        Path to directory of time series dataset (created with
        process_time_series.py).
    output_dir : pathlib.Path
        Path to desired output directory.
    """
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Process features series...")
    logging.info(f"time_series_dataset_dir: {time_series_dataset_dir}")
    logging.info(f"output_dir: {output_dir}")

    start_time = time.time()

    time_series = Dataset(time_series_dataset_dir)

    logging.info(f"Found {time_series.nb_patients} patients.")

    # Create the features dataset with basic infos
    create_new_dataset(directory=output_dir,
                       window_duration=time_series.window_duration,
                       train_window_offset=time_series.train_window_offset,
                       test_window_offset=time_series.test_window_offset,
                       overwrite=True)

    # Process each record and add them to the dataset
    for p in time_series.patients:
        patient_start_time = time.time()

        logging.info(f"Patient {p}/{time_series.nb_patients}")
        logging.info(f"Nb of records: "
                     f"{time_series.nb_records_per_patient[p]}")

        # Calculate features for each train record
        for _, record in time_series.split_leave_one_record_out(p):
            input_arr, labels_arr = time_series.get_data(record, "train")
            features_arr = _get_features(input_arr, SAMPLING_FREQUENCY)
            add_record_to_dataset(output_dir, str(p), "train",
                                  features_arr, labels_arr)

        # Calculate features for each test record
        for _, record in time_series.split_leave_one_record_out(p):
            input_arr, labels_arr = time_series.get_data(record, "test")
            features_arr = _get_features(input_arr, SAMPLING_FREQUENCY)
            add_record_to_dataset(output_dir, str(p), "test",
                                  features_arr, labels_arr)

        patient_duration = time.time() - patient_start_time
        logging.info(f"Pre-processed records from patient {p} "
                     f"in {patient_duration} s.")

    duration = time.time() - start_time
    logging.info(f"Completed processing of features data "
                 f"in {duration} seconds.")


def _get_features(arr: np.ndarray, sampling_freq: float) -> np.ndarray:
    """
    For each row in arr, calculate the following features:
        variance, skewness, kurtosis, median absolute deviation, line length,
        maximum, max power, mean power, power variance,
        theta band power, beta band power, gamma band power, epi index.

    Parameters
    ----------
    arr : np.ndarray
        Time-series array, where each row represents a window.
    sampling_freq : float
        Sampling frequency, in Hertz.

    Returns
    -------
    features : np.ndarray
        Array of calculated features.
    """
    nb_features = 15
    features = np.zeros((arr.shape[0], nb_features, arr.shape[2]))
    features[:, 0] = np.var(arr, axis=1)
    features[:, 1] = scipy.stats.skew(arr, axis=1)
    features[:, 2] = scipy.stats.kurtosis(arr, axis=1)
    features[:, 3] = scipy.stats.median_abs_deviation(arr, axis=1)
    features[:, 4] = np.sum(np.abs(np.diff(arr, axis=1)), axis=1)
    features[:, 5] = np.max(arr, axis=1)

    f, power = scipy.signal.welch(arr, fs=sampling_freq, window="hann", axis=1,
                                  scaling="spectrum")
    features[:, 6] = np.max(power, axis=1)
    features[:, 7] = np.mean(power, axis=1)
    features[:, 8] = np.var(power, axis=1)

    theta_band_st = _find_nearest(f, 4)[0]
    theta_band_end = _find_nearest(f, 8)[0]
    theta_band_power = power[:, theta_band_st:theta_band_end + 1].sum(axis=1)

    beta_band_st = _find_nearest(f, 13)[0]
    beta_band_end = _find_nearest(f, 30)[0]
    beta_band_power = power[:, beta_band_st:beta_band_end + 1].sum(axis=1)

    gamma_band_st = _find_nearest(f, 30)[0]
    gamma_band_end = _find_nearest(f, 45)[0]
    gamma_band_power = power[:, gamma_band_st:gamma_band_end + 1].sum(axis=1)

    lowfreq_band_st = _find_nearest(f, 3)[0]
    lowfreq_band_end = _find_nearest(f, 12)[0]
    lowfreq_band_power = power[:, lowfreq_band_st:lowfreq_band_end + 1].sum(axis=1)
    lowfreq_band_power = np.where(lowfreq_band_power == 0, 1e-15,
                                  lowfreq_band_power)
    epi_index = (beta_band_power + gamma_band_power) / lowfreq_band_power

    features[:, 9] = theta_band_power
    features[:, 10] = beta_band_power
    features[:, 11] = gamma_band_power
    features[:, 12] = epi_index
    return features


def _find_nearest(arr, value):
    arr = np.asarray(arr)
    idx = (np.abs(arr - value)).argmin()
    return idx, arr[idx]


if __name__ == '__main__':
    process_features()
