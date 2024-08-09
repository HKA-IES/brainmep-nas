# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import matplotlib.pyplot as plt
import optuna
import numpy as np
from optuna.visualization._hypervolume_history import (
    _get_hypervolume_history_info)

# import your own module
from .abstractmodelstudy import AbstractModelStudy


def plot_hypervolume(modelstudy: type[AbstractModelStudy],
                     ref_point: tuple, combine_curves: bool = False,
                     logx: bool = False, logy: bool = False) -> plt.Figure:
    """
    Plot the hypervolume with respect to trials for a given study.

    Parameters
    ----------
    modelstudy: type[AbstractModelStudy]
        Target study.
    ref_point: tuple
        Reference point for the hypervolume calculation.
    combine_curves: bool
        Whether to combine the curves of all outer folds into one curve showing
        the mean and standard deviation. If False, one curve per outer fold is
        displayed.
    logx: bool = False
        Apply log scale to x axis.
    logy: bool = False
        Apply log scale to y axis.
    """
    study_storage = f"sqlite:///{modelstudy.BASE_DIR}/study_storage.db"

    # Compute the hypervolume history from a private Optuna function
    # TODO: Use a public function or another package to compute the hypervolume
    #  history.
    hv_values_list = []
    n_trials_list = []
    for outer_fold in range(modelstudy.N_FOLDS):
        study_name = f"{modelstudy.NAME}_outer_fold_{outer_fold}"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)
        hv_history = _get_hypervolume_history_info(study, np.array(ref_point))
        hv_values_list.append(hv_history.values)
        n_trials_list.append(len(hv_history.trial_numbers))

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    if combine_curves:
        # Plot mean and standard deviation of the hypervolume history of all
        # outer folds.
        n_trials_min = np.min(n_trials_list)
        hv_values_mean = np.zeros(n_trials_min)
        hv_values_std = np.zeros(n_trials_min)
        for i in range(n_trials_min):
            values = [hv[i] for hv in hv_values_list]
            hv_values_mean[i] = np.mean(values)
            hv_values_std[i] = np.std(values)
        ax.step(np.arange(n_trials_min), hv_values_mean, label="mean")
        ax.fill_between(np.arange(n_trials_min),
                        hv_values_mean - hv_values_std,
                        hv_values_mean + hv_values_std,
                        step="pre", alpha=0.5, label="standard deviation")
    else:
        # Plot all outer fold hypervolume history as separate curves
        for outer_fold, hv_values in enumerate(hv_values_list):
            label = f"Outer fold {outer_fold}"
            ax.plot(np.arange(len(hv_values)), hv_values, label=label)

    ax.grid(True)
    ax.set_title("Hypervolume")
    ax.set_xlabel("Trials")
    ax.set_ylabel("Hypervolume")
    ax.legend()

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    plt.show()
    return fig


def plot_pareto_front(modelstudy: type[AbstractModelStudy], loop: str,
                      combine_curves: bool) -> plt.Figure:
    pass


def plot_parameters_importance(modelstudy: type[AbstractModelStudy],
                               metric: str) -> plt.Figure:
    pass


def plot_parameters_distribution(modelstudy: type[AbstractModelStudy],
                                 metric: str) -> plt.Figure:
    pass
