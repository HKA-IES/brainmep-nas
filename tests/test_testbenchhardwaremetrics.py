# -*- coding: utf-8 -*-

# import built-in module
import configparser
import pathlib
import tempfile

# import third-party modules
import pytest

# import your own module
from brainmepnas.testbenchhardwaremetrics import TestbenchHardwareMetrics


class TestTestbenchHardwareMetrics:
    """
    Create a copy of the file "remotetestbench_EXAMPLE.ini" under the name
    "remotetestbench.ini" and input the connection settings.

    Note: tests must be run with -s flag to avoid conflicts between pytest and
    the fabric.runner.Runner.
    """
    REMOTE_TESTBENCH_CONFIG = "remotetestbench.ini"

    # .tflite model comes from audio_example1 from Silicon Labs' Machine
    # Learning Toolkit.
    # Link: https://github.com/SiliconLabs/mltk/blob/master/mltk/models/examples/audio_example1.mltk.zip
    GOOD_TFLITE_MODEL = pathlib.Path("audio_example1.tflite")

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
        with tempfile.TemporaryDirectory() as temp_dir:
            hm = TestbenchHardwareMetrics(host=remote_testbench_config["credentials"]["host"],
                                          user=remote_testbench_config["credentials"]["user"],
                                          password=remote_testbench_config["credentials"]["password"],
                                          tflite_model_path=self.GOOD_TFLITE_MODEL,
                                          output_dir=pathlib.Path(temp_dir))

            assert 0 < hm.inference_energy < 0.001
            assert 0 < hm.inference_time < 1
            assert 0 < hm.memory_ram < 256000
            assert 0 < hm.memory_flash < 1500000
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
