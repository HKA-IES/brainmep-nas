# -*- coding: utf-8 -*-

# import built-in module
import dataclasses

# import third-party modules
try:
    from mltk.core import profile_model, ProfilingModelResults
except ImportError:
    _has_mltk = False
else:
    _has_mltk = True

# import your own module
from .hardwaremetrics import HardwareMetrics


@dataclasses.dataclass
class MltkHardwareMetrics(HardwareMetrics):
    """
    Store hardware metrics given by the Silicon Labs Machine Learning Toolkit
    (MLTK).

    Requires the optional dependency "silabs-mltk[full]". Install it by
    executing '$ pip install silabs-mltk[full]'.

    Attributes
    ----------
    profiling_results: mltk.core.ProfilingModelResults
        Results object from the MLTK profiler.
    """

    profiling_results: "ProfilingModelResults"

    def __init__(self, tflite_model_path: str):
        """
        Estimate the hardware performance of a TFLite model from the
        Silicon Labs MLTK Model Profiler.

        Note: loading the MLTK estimators in memory can take up to 200 seconds
        depending on the layers of the model. The problem seems to lie with
        onnx>=1.11. When the estimators are loaded once in a session, they
        do not need to be loaded again, so energy estimation runs much faster
        for further calls to this class.

        Parameters
        ----------
        tflite_model_path: str
            Path to .tflite file.
        """
        if not _has_mltk:
            raise ImportError("The mltk package is required to use this class."
                              " Install it by executing "
                              "'$ pip install silabs-mltk[full]'.",
                              name="silabs-mltk")

        self.profiling_results = profile_model(tflite_model_path,
                                               accelerator="MVP",
                                               return_estimates=True)

        self.ram_memory = self.profiling_results.runtime_memory_bytes
        self.flash_memory = self.profiling_results.flatbuffer_size
        self.time = self.profiling_results.time
        self.energy = self.profiling_results.energy
