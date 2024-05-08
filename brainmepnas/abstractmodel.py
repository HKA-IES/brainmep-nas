# -*- coding: utf-8 -*-

# import built-in module
import abc
import typing

# import third-party modules
import optuna

# import your own module


class AbstractModel(abc.ABC):
    params: typing.TypedDict

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def parametrize(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def parametrize_from_trial(self, trial: optuna.Trial):
        pass

    @abc.abstractmethod
    def save(self, file_path):
        pass

    @abc.abstractmethod
    def load(self, file_path):
        pass

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass

    @abc.abstractmethod
    def get_hardware_metrics(self):
        pass
