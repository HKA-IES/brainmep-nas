# -*- coding: utf-8 -*-

# import built-in module
from typing import List

# import third-party modules
import matplotlib.pyplot as plt
import optuna
import numpy as np
from optuna.visualization._hypervolume_history import (
    _get_hypervolume_history_info)
from optuna.importance import get_param_importances
import pandas as pd

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

    Returns
    -------
    fig: plt.Figure
        Hypervolume plot.
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


def plot_parameters_importance(modelstudy: type[AbstractModelStudy]) -> plt.Figure:
    """
    Plot the hypervolume with respect to trials for a given study.

    Parameters
    ----------
    modelstudy: type[AbstractModelStudy]
        Target study.

    Returns
    -------
    fig: plt.Figure
        Parameters importance plot.
    """
    study_storage = f"sqlite:///{modelstudy.BASE_DIR}/study_storage.db"

    param_importance_obj_1_list = []
    param_importance_obj_2_list = []
    for outer_fold in range(modelstudy.N_FOLDS):
        study_name = f"{modelstudy.NAME}_outer_fold_{outer_fold}"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)
        param_importance_obj_1 = get_param_importances(study,
                                                       target=lambda t: t.values[0])
        param_importance_obj_2 = get_param_importances(study,
                                                       target=lambda t: t.values[1])
        param_importance_obj_1_list.append(param_importance_obj_1)
        param_importance_obj_2_list.append(param_importance_obj_2)

    df_obj_1 = pd.DataFrame(param_importance_obj_1_list)
    df_obj_2 = pd.DataFrame(param_importance_obj_2_list)
    df_obj_1 = df_obj_1.reindex(sorted(df_obj_1.columns, reverse=True), axis=1)
    df_obj_2 = df_obj_2.reindex(sorted(df_obj_2.columns, reverse=True), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.barh(df_obj_1.columns, df_obj_1.mean(), height=0.4, align="edge",
            xerr=df_obj_1.std(),
            label=modelstudy.OBJ_1_METRIC,
            error_kw={'capsize': 3})
    ax.barh(df_obj_2.columns, df_obj_2.mean(), height=-0.4, align="edge",
            xerr=df_obj_2.std(),
            label=modelstudy.OBJ_2_METRIC,
            error_kw={'capsize': 3})

    ax.grid(True)
    ax.set_title(f"Parameters importance")
    ax.set_xlabel("Importance")
    ax.legend()

    xlim = ax.get_xlim()
    ax.set_xlim((0, xlim[1]))

    plt.tight_layout()
    plt.show()

    return fig


def plot_parameters_distribution(modelstudy: type[AbstractModelStudy]) -> List[plt.Figure]:
    """
    Plot the distribution of all parameters for a given study.

    Parameters
    ----------
    modelstudy: type[AbstractModelStudy]
        Target study.

    Returns
    -------
    fig: List[plt.Figure]
        List of parameters distribution plots.
    """
    study_storage = f"sqlite:///{modelstudy.BASE_DIR}/study_storage.db"

    figs = []

    for outer_fold in range(modelstudy.N_FOLDS):
        study_name = f"{modelstudy.NAME}_outer_fold_{outer_fold}"
        study = optuna.load_study(study_name=study_name,
                                  storage=study_storage)

        n_params = len(study.trials[0].params)
        if n_params > 1:
            n_cols = 2
        else:
            n_cols = 1
        n_rows = int(np.ceil(n_params / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2*n_rows))
        axs = axs.flatten()

        all_params = []
        for trial in study.trials:
            all_params.append(trial.params)

        df = pd.DataFrame(all_params)
        df.hist(ax=axs[:n_params])
        for ax in axs[:n_params]:
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
        fig.suptitle(f"Parameter distribution (outer fold {outer_fold})")
        plt.tight_layout()
        plt.show()

        figs.append(fig)

    return figs
