# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import pytest
import numpy as np

# import your own module
from brainmepnas import AccuracyMetrics, HardwareMetrics, CombinedMetrics


class TestCombinedMetrics:

    def test_init(self):
        """
        CombinedMetrics initializes properly when all params are specified.
        """
        # Initialize AccuracyMetrics and HardwareMetrics
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1, 0, 1, 0, 0])

        sample_duration = 4
        sample_offset = 4

        am = AccuracyMetrics(y_true, y_pred, sample_duration,
                             sample_offset, threshold=0.5)

        hm = HardwareMetrics(energy=1,
                             time=2,
                             ram_memory=3,
                             flash_memory=4)

        stimulation_energy = 10.0
        cm = CombinedMetrics(am, hm, stimulation_energy)

        assert cm.accuracy_metrics == am
        assert cm.hardware_metrics == hm
        assert cm.stimulation_energy == pytest.approx(stimulation_energy)

        assert cm.busy_cycle == pytest.approx(2/4)
        assert cm.inference_energy_per_hour == pytest.approx(3600/sample_offset*hm.energy)
        assert cm.event_tp_stimulation_energy_per_hour == pytest.approx(am.event_tp/am.total_duration*3600*stimulation_energy)
        assert cm.event_fp_stimulation_energy_per_hour == pytest.approx(am.event_fp/am.total_duration*3600*stimulation_energy)
        assert cm.event_stimulation_energy_per_hour == pytest.approx((am.event_tp+am.event_fp)/am.total_duration*3600*stimulation_energy)
        assert cm.undesired_energy_per_hour == pytest.approx(cm.event_fp_stimulation_energy_per_hour+cm.inference_energy_per_hour)

    def test_as_dict(self):
        """
        Function as_dict() returns a dictionary with the values.
        """
        # Initialize AccuracyMetrics and HardwareMetrics
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1, 0, 1, 0, 0])

        sample_duration = 4
        sample_offset = 4

        am = AccuracyMetrics(y_true, y_pred, sample_duration,
                             sample_offset, threshold=0.5)

        hm = HardwareMetrics(energy=1,
                             time=2,
                             ram_memory=3,
                             flash_memory=4)

        cm = CombinedMetrics(am, hm, stimulation_energy=10)

        expected_dict = {"stimulation_energy": cm.stimulation_energy,
                         "busy_cycle": cm.busy_cycle,
                         "inference_energy_per_hour": cm.inference_energy_per_hour,
                         "event_tp_stimulation_energy_per_hour": cm.event_tp_stimulation_energy_per_hour,
                         "event_fp_stimulation_energy_per_hour": cm.event_fp_stimulation_energy_per_hour,
                         "event_stimulation_energy_per_hour": cm.event_stimulation_energy_per_hour,
                         "undesired_energy_per_hour": cm.undesired_energy_per_hour}

        assert cm.as_dict() == expected_dict
