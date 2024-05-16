# -*- coding: utf-8 -*-

# import built-in module
import dataclasses

# import third-party modules
import numpy as np
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
    nnresults: seizuredetectiontestbench.NNResults
        Results object from the seizure detection testbench.
    """

    nnresults: "NNResults"

    def __init__(self, host: str, user: str, password: str,
                 tflite_model_path: str):
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
        tflite_model_path: str
            Path to .tflite file.
        """
        rt = RemoteTestbench(host, user, password)
        rt.begin_test_tflite_model(tflite_model_path)

        self.nnresults = rt.get_results_test_tflite_model()
        self.energy = np.mean(self.nnresults.total_energy)
        self.time = np.mean(self.nnresults.total_duration)
        self.ram_memory = self.nnresults.memory_ram
        self.flash_memory = self.nnresults.memory_flash
