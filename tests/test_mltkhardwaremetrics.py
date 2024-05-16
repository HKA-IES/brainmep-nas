# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import pytest

# import your own module
from brainmepnas.mltkhardwaremetrics import MltkHardwareMetrics


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

        assert hm.clock_frequency == pytest.approx(78e6)
        assert hm.tflite_size == 16000
        assert hm.runtime_memory_size == 24388
        assert hm.inference_ops == 1378052
        assert hm.inference_macs == 671184
        assert hm.inference_cpu_cycles == 264240
        assert hm.inference_accelerator_cycles == 628310
        assert hm.inference_cpu_utilization == pytest.approx(39.58, abs=0.01)
        assert hm.j_per_op == pytest.approx(1.09e-10, abs=0.01e-10)
        assert hm.j_per_mac == pytest.approx(2.23e-10, abs=0.01e-10)
        assert hm.op_per_s == pytest.approx(161e6, abs=1e6)
        assert hm.mac_per_s == pytest.approx(78e6, abs=1e6)
        assert hm.inference_per_sec == pytest.approx(116, abs=1)
        assert hm.profiling_results is not None

    def test_bad_tflite_model(self):
        """
        Error is raised if the tflite model is not valid.
        """
        with pytest.raises(ValueError):
            MltkHardwareMetrics("bad")
