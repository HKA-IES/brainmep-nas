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

    @pytest.mark.usefixtures('hide_mltk_package')
    def test_mltk_missing(self):
        """
        ImportError is raised if mltk package is missing.
        """
        with pytest.raises(ImportError):
            MltkHardwareMetrics(self.GOOD_TFLITE_MODEL)

    def test_good_tflite_model(self):
        """
        Appropriate results are obtained for a valid tflite model.
        """
        hm = MltkHardwareMetrics(self.GOOD_TFLITE_MODEL)

    def test_bad_tflite_model(self):
        """
        Error is raised if the tflite model is not valid.
        """
        with pytest.raises(ValueError):
            MltkHardwareMetrics("bad")
