# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import pytest
import numpy as np

# import your own module
from brainmepnas import AccuracyMetrics, HardwareMetrics, CombinedMetrics


class TestCombinedMetrics:

    def test_init_without_stimulation_energy(self):
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

        inference_energy = 1
        inference_time = 2

        hm = HardwareMetrics(inference_time, inference_energy)

        cm = CombinedMetrics(am, hm)

        assert cm.accuracy_metrics == am
        assert cm.hardware_metrics == hm
        assert cm.stimulation_energy is None

        assert cm.inference_duty_cycle == pytest.approx(inference_time /
                                                        sample_offset)
        assert cm.total_inference_energy == pytest.approx(len(y_true) *
                                                          inference_energy)
        assert cm.inference_energy_per_hour == pytest.approx((3600 / sample_offset) *
                                                             inference_energy)

        assert cm.event_tp_total_stimulation_energy is None
        assert cm.event_tp_stimulation_energy_per_hour is None
        assert cm.event_fp_total_stimulation_energy is None
        assert cm.event_fp_stimulation_energy_per_hour is None
        assert cm.event_total_stimulation_energy is None
        assert cm.event_stimulation_energy_per_hour is None
        assert cm.total_undesired_energy is None
        assert cm.undesired_energy_per_hour is None

    def test_init_with_stimulation_energy(self):
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

        inference_energy = 1
        inference_time = 2

        hm = HardwareMetrics(inference_time, inference_energy)

        stimulation_energy = 10.0
        cm = CombinedMetrics(am, hm, stimulation_energy)

        assert cm.accuracy_metrics == am
        assert cm.hardware_metrics == hm
        assert cm.stimulation_energy == pytest.approx(stimulation_energy)

        assert cm.inference_duty_cycle == pytest.approx(inference_time/sample_offset)
        assert cm.total_inference_energy == pytest.approx(len(y_true) * inference_energy)
        assert cm.inference_energy_per_hour == pytest.approx(3600 / sample_offset * inference_energy)

        assert cm.event_tp_total_stimulation_energy == pytest.approx(am.event_tp *
                                                                     stimulation_energy)
        assert cm.event_tp_stimulation_energy_per_hour == pytest.approx(am.event_tp *
                                                                        (3600 / am.total_duration) *
                                                                        stimulation_energy)
        assert cm.event_fp_total_stimulation_energy == pytest.approx(am.event_fp *
                                                                     stimulation_energy)
        assert cm.event_fp_stimulation_energy_per_hour == pytest.approx(am.event_fp *
                                                                        (3600 / am.total_duration) *
                                                                        stimulation_energy)
        assert cm.event_total_stimulation_energy == pytest.approx((am.event_tp + am.event_fp) *
                                                                     stimulation_energy)
        assert cm.event_stimulation_energy_per_hour == pytest.approx((am.event_tp + am.event_fp) *
                                                                        (3600 / am.total_duration) *
                                                                        stimulation_energy)
        assert cm.total_undesired_energy == pytest.approx(cm.total_inference_energy +
                                                          cm.event_fp_total_stimulation_energy)
        assert cm.undesired_energy_per_hour == pytest.approx(cm.inference_energy_per_hour +
                                                             cm.event_fp_stimulation_energy_per_hour)

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

        inference_energy = 1
        inference_time = 2

        hm = HardwareMetrics(inference_time, inference_energy)

        stimulation_energy = 10.0
        cm = CombinedMetrics(am, hm, stimulation_energy)

        expected_dict = {"stimulation_energy": cm.stimulation_energy,
                         "inference_duty_cycle": cm.inference_duty_cycle,
                         "total_inference_energy": cm.total_inference_energy,
                         "inference_energy_per_hour": cm.inference_energy_per_hour,
                         "event_tp_total_stimulation_energy": cm.event_tp_total_stimulation_energy,
                         "event_tp_stimulation_energy_per_hour": cm.event_tp_stimulation_energy_per_hour,
                         "event_fp_total_stimulation_energy": cm.event_fp_total_stimulation_energy,
                         "event_fp_stimulation_energy_per_hour": cm.event_fp_stimulation_energy_per_hour,
                         "event_total_stimulation_energy": cm.event_total_stimulation_energy,
                         "event_stimulation_energy_per_hour": cm.event_stimulation_energy_per_hour,
                         "total_undesired_energy": cm.total_undesired_energy,
                         "undesired_energy_per_hour": cm.undesired_energy_per_hour}

        assert cm.as_dict() == expected_dict
