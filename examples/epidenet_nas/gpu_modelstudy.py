# -*- coding: utf-8 -*-

# import built-in module
import pathlib

# import third-party modules

# import your own module
from basemodelstudy import BaseModelStudy


class GPUModelStudy(BaseModelStudy):
    N_PARALLEL_GPU_JOBS = 2
    N_PARALLEL_CPU_JOBS = 1
    GET_ACCURACY_METRICS_USE_GPU = True
    GET_HARDWARE_METRICS_USE_GPU = False
    NAME = "gpu_modelstudy"
    BASE_DIR = pathlib.Path("gpu_modelstudy/")
    THIS_FILE = pathlib.Path(__file__)


if __name__ == "__main__":
    GPUModelStudy.cli_entry_point()
