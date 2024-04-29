# -*- coding: utf-8 -*-

# import built-in module
import builtins

# import third-party modules
import pytest
import numpy as np

# import your own module
from brainmepnas import MltkHardwareMetrics


class TestMltkHardwareMetrics:
    # .tflite model comes from audio_example1 from Silicon Labs' Machine
    # Learning Toolkit.
    # Link: https://github.com/SiliconLabs/mltk/blob/master/mltk/models/examples/audio_example1.mltk.zip
    GOOD_TFLITE_MODEL = "audio_example1.tflite"

    def test_good_tflite_model(self):
        """
        Appropriate results are obtained for a valid tflite model.

        Note: due to an issue with onnx, this test takes ~200 seconds to run.
        """
        hm = MltkHardwareMetrics(self.GOOD_TFLITE_MODEL)

        assert 0 < hm.energy < 0.001
        assert 0 < hm.time < 1
        assert 0 < hm.ram_memory < 256000
        assert 0 < hm.flash_memory < 1500000
        assert hm.profiling_results is not None

    def test_bad_tflite_model(self):
        """
        Error is raised if the tflite model is not valid.
        """
        with pytest.raises(ValueError):
            MltkHardwareMetrics("bad")
