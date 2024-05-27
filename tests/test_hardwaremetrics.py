# -*- coding: utf-8 -*-

# import built-in module
import tempfile
import pickle

# import third-party modules
import pytest

# import your own module
from brainmepnas import HardwareMetrics


class TestHardwareMetrics:

    def test_init(self):
        """
        HardwareMetrics initializes properly when all params are specified.
        """
        hm = HardwareMetrics(inference_time=1,
                             inference_energy=2)

        assert hm.inference_time == 1
        assert hm.inference_energy == 2

    def test_init_missing_values(self):
        """
        TypeError is raised if values are missing on initialization.
        """
        with pytest.raises(TypeError):
            HardwareMetrics()

    def test_as_dict(self):
        """
        Function as_dict() returns a dictionary with the values.
        """
        hm = HardwareMetrics(inference_time=1,
                             inference_energy=2)

        expected_dict = {"inference_time": 1,
                         "inference_energy": 2}

        assert hm.as_dict() == expected_dict

    def test_pickle(self):
        """
        HardwareMetrics should be pickleable.
        """
        hm = HardwareMetrics(inference_time=1,
                             inference_energy=2)
        with tempfile.TemporaryFile() as tmpfile:
            pickle.dump(hm, tmpfile)
