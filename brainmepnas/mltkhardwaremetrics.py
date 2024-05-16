# -*- coding: utf-8 -*-

# import built-in module
import dataclasses

# import third-party modules
try:
    from mltk.core import profile_model, ProfilingModelResults
except ImportError:
    raise ImportError("The mltk package is required to use this class. Install"
                      " it by executing '$ pip install silabs-mltk[full]'.",
                      name="silabs-mltk")

# import your own module
from .hardwaremetrics import HardwareMetrics


@dataclasses.dataclass
class MltkHardwareMetrics(HardwareMetrics):
    """
    Store hardware metrics given by the Silicon Labs Machine Learning Toolkit
    (MLTK).

    Requires the optional dependency "silabs-mltk[full]". Install it by
    executing '$ pip install silabs-mltk[full]'.

    Note: the model profiler provides estimated values for the inference_time
    and inference_energy. inference_current_avg and inference_power_avg are
    estimated by assuming that the profile operates with a voltage of 1.8 V,
    which seems to be validated by experimental measurements, but has not
    been formally stated by Silicon Labs.

    Attributes
    ----------
    clock_frequency: float
        Clock frequency in hertz (Hz).
    tflite_size: int
        Size of the TFLite model in memory (flash), in bytes (B).
    runtime_memory_size: int
        Runtime memory (RAM) required to perform an inference, in bytes (B).
    inference_ops: int
        Number of operations for one inference.
    inference_macs: int
        Number of multiply-accumulate operations for one inference.
    inference_cpu_cycles: int
        Number of CPU cycles for one inference.
    inference_accelerator_cycles: int
        Number of accelerator cycles for one inference.
    inference_cpu_utilization: float
        Ratio of the CPU used for one inference, calculated as follows:
            cpu_utilization = cpu_cycles / (time*clock_frequency)
    j_per_op: float
        Energy per operation, in joules (J).
    j_per_mac: float
        Energy per multiply-accumulate operation, in joules (J).
    op_per_s: float
        Operations per second, in 1/seconds (1/s).
    mac_per_s: float
        Multiply-accumulate operations per second, in 1/seconds (1/s).
    inference_per_sec: float
        Number of inferences per second, in 1/seconds (1/s).
    profiling_results: mltk.core.ProfilingModelResults
        Results object from the MLTK profiler.
    """

    clock_frequency: float

    tflite_size: int
    runtime_memory_size: int

    inference_ops: int
    inference_macs: int
    inference_cpu_cycles: int
    inference_accelerator_cycles: int
    inference_cpu_utilization: float
    j_per_op: float
    j_per_mac: float
    op_per_s: float
    mac_per_s: float
    inference_per_sec: float

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
        self.profiling_results = profile_model(tflite_model_path,
                                               accelerator="MVP",
                                               return_estimates=True)
        summary = self.profiling_results.get_summary(exclude_null=False,
                                                     full_summary=True)

        self.inference_time = summary["time"]
        self.inference_energy = summary["energy"]

        self.clock_frequency = float(summary["cpu_clock_rate"])
        self.tflite_size = int(summary["tflite_size"])
        self.runtime_memory_size = int(summary["runtime_memory_size"])
        self.inference_ops = int(summary["ops"])
        self.inference_macs = int(summary["macs"])
        self.inference_cpu_cycles = int(summary["cpu_cycles"])
        self.inference_accelerator_cycles = int(summary["accelerator_cycles"])
        self.inference_cpu_utilization = summary["cpu_utilization"]
        self.j_per_op = summary["j_per_op"]
        self.j_per_mac = summary["j_per_mac"]
        self.op_per_s = summary["op_per_s"]
        self.mac_per_s = summary["mac_per_s"]
        self.inference_per_sec = summary["inf_per_sec"]
