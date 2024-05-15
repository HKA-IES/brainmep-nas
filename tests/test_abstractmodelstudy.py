# -*- coding: utf-8 -*-

# import built-in module
import configparser

# import third-party modules
import pytest

# import your own module
from brainmepnas import AbstractModelStudy
from dummymodelstudy import DummyModelStudy


class TestAbstractModelStudy:
    def test_abstract_class_not_instantiable(self):
        with pytest.raises(RuntimeError):
            AbstractModelStudy()

    def test_abstract_class_method_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._sample_search_space(None)
        with pytest.raises(NotImplementedError):
            AbstractModelStudy.get_accuracy_metrics(None, None)
        with pytest.raises(NotImplementedError):
            AbstractModelStudy.get_hardware_metrics(None, None)
