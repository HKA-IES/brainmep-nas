# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import matplotlib.pyplot as plt

# import your own module
from brainmepnas.visualization import plot_hypervolume
from cpu_modelstudy import CPUModelStudy

if __name__ == "__main__":
    print("ok")
    plot_hypervolume(CPUModelStudy, (0, 1), combine_curves=True)
