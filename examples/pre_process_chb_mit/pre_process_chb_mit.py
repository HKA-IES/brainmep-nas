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

# import your own module
from brainmepnas.dataset import create_new_dataset, add_record_to_dataset

# Fixed parameters
TRAIN_WINDOW_LENGTH = 4     # seconds
TRAIN_WINDOW_OVERLAP = 0    # seconds
TEST_WINDOW_LENGTH = TRAIN_WINDOW_LENGTH    # seconds
TEST_WINDOW_OVERLAP = 2     # seconds
CHANNELS = ["F7-T7", "T7-P7", "F8-T8", "T8-P8-0"]

@click.command()
@click.option("-i", "--input",
              type=click.Path(exists=True, dir_okay=True, path_type=pathlib.Path),
              required=True,
              help="Path to raw data directory.")
@click.option("-o", "--output",
              type=click.Path(dir_okay=True, path_type=pathlib.Path),
              required=True,
              help="Path to output directory.")
def pre_process_chb_mit(raw_data_dir: pathlib.Path, output_dir: pathlib.Path):
    """
    Pre-process CHB-MIT Scalp EEG data and format the output as a Dataset.

    Parameters
    ----------
    raw_data_dir : pathlib.Path
        Path to raw CHB-MIT Scalp EEG data.
    output_dir : pathlib.Path
        Path to desired output directory.
    """
    logging.info("Generating dataset...")
    logging.info(f"raw_data_dir: {raw_data_dir}")
    logging.info(f"output_dir: {output_dir}")

    start_time = time.time()

    # Get all the files for which there is at least one seizure.
    records_with_seizures_file_path = raw_data_dir / "RECORDS-WITH-SEIZURES"
    records_with_seizures = dict()  # key is patient, value is list of paths.

    file = open(records_with_seizures_file_path, "r")
    for line in file:
        if line == "\n":
            continue

        # Remove \n
        line = line[:-1]

        try:
            records_with_seizures[int(line[3:5])].append(line)
        except KeyError:
            records_with_seizures[int(line[3:5])] = [line]
    file.close()

    nb_patients = len(records_with_seizures)
    logging.info(f"Found {nb_patients} patients.")

    # Create the dataset with basic informations
    sample_record_path = list(records_with_seizures.values())[0][0]
    sample_edf = mne.io.read_raw_edf(raw_data_dir / sample_record_path)
    sampling_frequency = sample_edf.info["sfreq"]
    train_window_length = TRAIN_WINDOW_LENGTH * sampling_frequency
    train_window_overlap = TRAIN_WINDOW_OVERLAP * sampling_frequency
    test_window_length = TEST_WINDOW_LENGTH * sampling_frequency
    test_window_overlap = TEST_WINDOW_OVERLAP * sampling_frequency

    create_new_dataset(directory=output_dir,
                       sampling_frequency=sampling_frequency,
                       train_window_overlap=train_window_overlap,
                       test_window_overlap=test_window_overlap,
                       overwrite=True)

    # Process each record and add them to the dataset
    for p in records_with_seizures:
        patient_start_time = time.time()

        logging.info(f"Patient {p}/{nb_patients}")
        logging.info(f"Nb of records with at least one seizure: "
                     f"{len(records_with_seizures[p])}")

        # Load data from .edf
        for record_path in records_with_seizures[p]:
            try:
                r = int(record_path[12:14])
            except ValueError:
                r = int(record_path[13:15])
            logging.info(f"Processing record {record_path}")

            edf = mne.io.read_raw_edf(raw_data_dir / record_path)
            try:
                edf = edf.pick(CHANNELS)
            except ValueError:
                # Expected channels are not (all) present. Skip this record.
                logging.warning(f"Record {record_path}: all expected channels are not present. Skipping this record.")
                continue
            edf.load_data()  # Accelerates pre-processing steps

            # Load seizure annotations from .edf.seizures
            with open(raw_data_dir / (record_path + ".seizures"), "rb") as f:
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

                edf.set_annotations(mne.Annotations(start_times[1:],
                                                    durations[1:],
                                                    "ictal"))

            # Signal processing on all the relevant channels.

            # Butterworth filter, order 5, 0.5 Hz to 50 Hz
            iir_params = dict(order=5, ftype="butterworth")
            edf.filter(l_freq=0.5, h_freq=50.0, method="iir",
                       iir_params=iir_params)

            # Prepare training set
            logging.info("\tPreparing training set.")
            train_epochs = mne.make_fixed_length_epochs(edf,
                                                        duration=TRAIN_WINDOW_LENGTH,
                                                        overlap=TRAIN_WINDOW_OVERLAP)
            train_input = np.array(train_epochs.get_data(),
                                   dtype=np.float16)

            train_scaler = mne.decoding.Scaler(scalings="mean")
            train_input_scaled = train_scaler.fit_transform(train_input)

            # we want channels last
            train_input_scaled = np.swapaxes(train_input_scaled, 1, 2)

            train_labels = np.zeros((train_input_scaled.shape[0], 1),
                                    dtype=np.float16)

            nb_epochs_ictal = 0
            for i, annotations in enumerate(train_epochs.get_annotations_per_epoch()):
                if len(annotations) > 0:
                    logging.debug(f"\tEpoch {i} is ictal.")
                    nb_epochs_ictal += 1
                    train_labels[i] = 1

            logging.info(f"\t{nb_epochs_ictal}/{len(train_epochs)} epochs "
                         f"are ictal")

            add_record_to_dataset(output_dir, str(p), "train",
                                  train_input_scaled, train_labels)

            # Prepare test set
            logging.info("\tPreparing test set.")
            test_epochs = mne.make_fixed_length_epochs(edf,
                                                       duration=TEST_WINDOW_LENGTH,
                                                       overlap=TEST_WINDOW_OVERLAP)
            test_input = np.array(test_epochs.get_data(), dtype=np.float16)

            test_scaler = mne.decoding.Scaler(scalings="mean")
            test_input_scaled = test_scaler.fit_transform(test_input)

            # we want channels last
            test_input_scaled = np.swapaxes(test_input_scaled, 1, 2)

            test_labels = np.zeros((test_input_scaled.shape[0], 1),
                                   dtype=np.float16)
            nb_epochs_ictal = 0
            for i, annotations in enumerate(
                    test_epochs.get_annotations_per_epoch()):
                if len(annotations) > 0:
                    logging.debug(f"Epoch {i} is ictal.")
                    nb_epochs_ictal += 1
                    test_labels[i] = 1
            logging.info(f"\t{nb_epochs_ictal}/{len(test_epochs)} epochs "
                         f"are ictal")

            add_record_to_dataset(output_dir, str(p), "test",
                                  test_input_scaled, test_labels)

        patient_duration = time.time() - patient_start_time
        logging.info(f"Pre-processed records from patient {p} in {patient_duration} s.")

    duration = time.time() - start_time
    logging.info(f"Dataset generation completed in {duration} seconds.")


if __name__ == '__main__':
    pre_process_chb_mit()
