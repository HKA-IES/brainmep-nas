# -*- coding: utf-8 -*-

# import built-in module
import configparser

# import third-party modules
import pytest

# import your own module
from brainmepnas.testbenchhardwaremetrics import TestbenchHardwareMetrics


class TestTestbenchHardwareMetrics:
    """
    Create a copy of the file "remotetestbench_EXAMPLE.ini" under the name
    "remotetestbench.ini" and input the connection settings.
    """
    REMOTE_TESTBENCH_CONFIG = "remotetestbench.ini"

    # .tflite model comes from audio_example1 from Silicon Labs' Machine
    # Learning Toolkit.
    # Link: https://github.com/SiliconLabs/mltk/blob/master/mltk/models/examples/audio_example1.mltk.zip
    GOOD_TFLITE_MODEL = "audio_example1.tflite"

    # TODO: Consider mocking RemoteTestbench.

    @pytest.fixture
    def remote_testbench_config(self):
        config = configparser.ConfigParser()
        config.read(self.REMOTE_TESTBENCH_CONFIG)
        return config

    def test_good_tflite_model(self, remote_testbench_config):
        """
        Appropriate results are obtained for a valid tflite model.
        """
        hm = TestbenchHardwareMetrics(host=remote_testbench_config["credentials"]["host"],
                                      user=remote_testbench_config["credentials"]["user"],
                                      password=remote_testbench_config["credentials"]["password"],
                                      tflite_model_path=self.GOOD_TFLITE_MODEL)

        assert 0 < hm.energy < 0.001
        assert 0 < hm.time < 1
        assert 0 < hm.ram_memory < 256000
        assert 0 < hm.flash_memory < 1500000
        assert hm.nnresults is not None

    def test_bad_tflite_model(self, remote_testbench_config):
        """
        Error is raised if the tflite model is not valid.
        """
        with pytest.raises(FileNotFoundError):
            TestbenchHardwareMetrics(
                host=remote_testbench_config["credentials"]["host"],
                user=remote_testbench_config["credentials"]["user"],
                password=remote_testbench_config["credentials"]["password"],
                tflite_model_path="bad")
