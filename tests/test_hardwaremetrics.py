# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import pytest

# import your own module
from brainmepnas import HardwareMetrics


class TestHardwareMetrics:

    def test_init(self):
        """
        HardwareMetrics initializes properly when all params are specified.
        """
        hm = HardwareMetrics(energy=1,
                             time=2,
                             ram_memory=3,
                             flash_memory=4)

        assert hm.energy == 1
        assert hm.time == 2
        assert hm.ram_memory == 3
        assert hm.flash_memory == 4

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
        hm = HardwareMetrics(energy=1,
                             time=2,
                             ram_memory=3,
                             flash_memory=4)

        expected_dict = {"energy": 1,
                         "time": 2,
                         "ram_memory": 3,
                         "flash_memory": 4}

        assert hm.as_dict() == expected_dict
