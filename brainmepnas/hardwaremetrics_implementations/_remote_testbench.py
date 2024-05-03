# -*- coding: utf-8 -*-

# import built-in module
import dataclasses

# import third-party modules
import numpy as np
try:
    from seizuredetectiontestbench import RemoteTestbench, NNResults
except ImportError:
    _has_seizuredetectiontestbench = False
else:
    _has_seizuredetectiontestbench = True

# import your own module
from ..hardwaremetrics import HardwareMetrics


@dataclasses.dataclass
class TestbenchHardwareMetrics(HardwareMetrics):
    """
    Store hardware metrics given by the Seizure Detection Testbench.
    
    Requires the optional dependency "seizure-detection-testbench". Install it
    by executing
    '$ pip install git+https://{USERNAME}@bitbucket.org/precisis/seizure_detection_testbench.git'.
    """

    nnresults: "NNResults"

    def __init__(self, host: str, user: str, password: str,
                 tflite_model_path: str):
        """
        :param host: IP address of Testbench.
        :param user: User
        :param password: Password.
        :param tflite_model_path: path to .tflite file.
        """
        if not _has_seizuredetectiontestbench:
            raise ImportError("The seizuredetectiontestbench package is "
                              "required to use this class."
                              " Install it by executing "
                              "'$ pip install git+https://{USERNAME}@bitbucket.org/precisis/seizure_detection_testbench.git'.",
                              name="seizuredetectiontestbench")

        rt = RemoteTestbench(host, user, password)
        rt.begin_test_tflite_model(tflite_model_path)

        self.nnresults = rt.get_results_test_tflite_model()
        self.energy = np.mean(self.nnresults.total_energy)
        self.time = np.mean(self.nnresults.total_duration)
        self.ram_memory = self.nnresults.memory_ram
        self.flash_memory = self.nnresults.memory_flash
