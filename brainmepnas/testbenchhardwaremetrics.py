# -*- coding: utf-8 -*-

# import built-in module
import dataclasses
import pathlib
import warnings
from typing import Optional, Union

# import third-party modules
import numpy as np
import pandas as pd
try:
    from seizuredetectiontestbench import RemoteTestbench, NNResults
except ImportError:
    raise ImportError("The seizuredetectiontestbench package is required to "
                      "use this class. Install it by executing "
                      "'$ pip install git+https://{USERNAME}@bitbucket.org/precisis/seizure_detection_testbench.git'.",
                      name="seizuredetectiontestbench")

# import your own module
from .hardwaremetrics import HardwareMetrics


@dataclasses.dataclass
class TestbenchHardwareMetrics(HardwareMetrics):
    """
    Store hardware metrics given by the Seizure Detection Testbench.
    
    Requires the optional dependency "seizure-detection-testbench". Install it
    by executing
    '$ pip install git+https://{USERNAME}@bitbucket.org/precisis/seizure_detection_testbench.git'.

    Note: The Seizure Detection Testbench is a private tool of the BrainMEP
    project and there are currently no plans of making it publicly available.

    Attributes
    ----------
    memory_ram: int
        Estimated RAM use, in bytes (B).
    memory_flash: int
        Flash memory used, in bytes (B).
    nnresults: seizuredetectiontestbench.NNResults
        Results object from the seizure detection testbench.
    """

    memory_ram: int
    memory_flash: int

    nnresults: "NNResults"

    def __init__(self, host: str, user: str, password: str,
                 tflite_model_path: Union[str, pathlib.Path],
                 output_dir: Union[str, pathlib.Path]):
        """
        Estimate the hardware performance of a TFLite model from the
        Seizure Detection Testbench.

        Note: The Seizure Detection Testbench is a private tool of the BrainMEP
        project and there are currently no plans of making it publicly
        available.

        Parameters
        ----------
        host: str
            IP address of remote testbench.
            Example: "192.0.2.1"
        user: str
            User of remote testbench.
        password: str
            Password of remote testbench for user in plaintext. Make sure to
            never write the password in any file which could be made public
            through external or public versioning tools.
        tflite_model_path: Union[str, pathlib.Path]
            Path to .tflite file.
        output_dir: Union[str, pathlib.Path]
            Directory to store output traces from the remote testbench.

        Raises
        ------
        RuntimeError
            If the measured values are all outliers compared to the expected
            inference duration.
        """
        if isinstance(tflite_model_path, str):
            tflite_model_path = pathlib.Path(tflite_model_path)

        rt = RemoteTestbench(host, user, password)
        rt.begin_test_tflite_model(tflite_model_path)

        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)
        nnresults = rt.get_results_test_tflite_model(str(output_dir.resolve()))

        # The testbench returns values for a certain number of inferences.
        # Due to instabilities with the testbench itself (which should be
        # eventually corrected), some inferences randomly run badly,
        # leading to outlier energy and/or duration measurements.

        # A true reference is the duration measured on the microcontroller and
        # outputed directly to the iostream.
        #   These logs are stored in mcu_iostream_parsed.csv
        energy_measurement_log_path = output_dir / "mcu_iostream_parsed.csv"
        df_mcu_iostream = pd.read_csv(energy_measurement_log_path)
        expected_execution_time = df_mcu_iostream[df_mcu_iostream["layer_name"] == "ALL"][
            "execution_time"].mean()

        # We expect real measured durations to be within +- 10% of
        # expected_execution_time.
        idx_to_ignore = []
        for id, duration in enumerate(nnresults.total_duration):
            if not 0.9 * expected_execution_time < duration < 1.1 * expected_execution_time:
                idx_to_ignore.append(id)
        if len(idx_to_ignore) == nnresults.nb_repetitions:
            raise RuntimeError("All iterations were excluded.")
        elif len(idx_to_ignore) > 0:
            warnings.warn(f"{len(idx_to_ignore)} iterations were excluded "
                          f"due to unexpectedly long or short inference time.")

        energy_values_subset = np.delete(nnresults.total_energy, idx_to_ignore)
        duration_values_subset = np.delete(nnresults.total_duration, idx_to_ignore)

        self.inference_energy = np.mean(energy_values_subset)
        self.inference_time = np.mean(duration_values_subset)
        self.memory_ram = nnresults.memory_ram
        self.memory_flash = nnresults.memory_flash
        self.nnresults = nnresults
