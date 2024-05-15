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
        with pytest.raises(TypeError):
            AbstractModelStudy()

    def test_abstract_class_method_not_implemented(self):
        with pytest.raises(NotImplementedError):
            AbstractModelStudy._sample_search_space(None)
            AbstractModelStudy._run_parallel_to_all_folds(None)
