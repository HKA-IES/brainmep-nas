# -*- coding: utf-8 -*-

# import built-in module
import pathlib

# import third-party modules

# import your own module
from basemodelstudy import BaseModelStudy


class CPUModelStudy(BaseModelStudy):
    N_PARALLEL_GPU_JOBS = 0
    N_PARALLEL_CPU_JOBS = 5
    GET_ACCURACY_METRICS_USE_GPU = False
    GET_HARDWARE_METRICS_USE_GPU = False
    NAME = "cpu_modelstudy"
    BASE_DIR = pathlib.Path("cpu_modelstudy/")
    THIS_FILE = pathlib.Path(__file__)


if __name__ == "__main__":
    CPUModelStudy.cli_entry_point()
