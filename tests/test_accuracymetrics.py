# -*- coding: utf-8 -*-

# import built-in module
import tempfile
import pickle

# import third-party modules
import pytest
import numpy as np

# import your own module
from brainmepnas import AccuracyMetrics


class TestAccuracyMetrics:
    """
    The calculation of metrics should be very robust in order to avoid wrong
    results and analyzes/conclusions down the line. For this reason, we aim for
    100% coverage of accuracymetrics.py. Redundancy between tests is not a
    problem.

    The test cases are illustrated in test_accuracymetrics.ods
    """

    # TODO: Add tests for invalid arguments

    def test_threshold_fixed(self):
        """
        Verify that the fixed threshold is applied properly by comparing two
        y_pred arrays which can be made equivalent depending on the chosen
        threshold.
        """
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred_1 = np.array([0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9,
                             0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9, 0.9,
                             0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                             0.9, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.0])
        y_pred_2 = np.array([0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7,
                             0.0, 0.0, 0.7, 0.7, 0.7, 0.0, 0.7, 0.7, 0.7, 0.7,
                             0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                             0.7, 0.7, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.7, 0.0, 0.0])
        sample_duration = 1
        sample_offset = 1

        am_1 = AccuracyMetrics(y_true, y_pred_1, sample_duration,
                               sample_offset, threshold=0.8)
        am_2 = AccuracyMetrics(y_true, y_pred_2, sample_duration,
                               sample_offset, threshold=0.8)
        assert am_1.sample_tp != am_2.sample_tp
        assert am_1.sample_tn != am_2.sample_tn
        assert am_1.sample_fp != am_2.sample_fp
        assert am_1.sample_fn != am_2.sample_fn

        am_3 = AccuracyMetrics(y_true, y_pred_1, sample_duration,
                               sample_offset, threshold=0.5)
        am_4 = AccuracyMetrics(y_true, y_pred_2, sample_duration,
                               sample_offset, threshold=0.5)
        assert am_3.sample_tp == am_4.sample_tp
        assert am_3.sample_tn == am_4.sample_tn
        assert am_3.sample_fp == am_4.sample_fp
        assert am_3.sample_fn == am_4.sample_fn

        assert am_1.threshold_method == "fixed"

    def test_threshold_max_f_score(self):
        """
        Verify that the threshold is calculated correctly with the max_f_score
        method. In the two simple cases below, the threshold should be <= 0.9
        and <= 0.7 respectively, and the metrics should be equivalent.
        """
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred_1 = np.array([0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9,
                             0.0, 0.0, 0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9, 0.9,
                             0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                             0.9, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.0])
        y_pred_2 = np.array([0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7,
                             0.0, 0.0, 0.7, 0.7, 0.7, 0.0, 0.7, 0.7, 0.7, 0.7,
                             0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                             0.7, 0.7, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.7, 0.0, 0.0])
        sample_duration = 1
        sample_offset = 1

        am_1 = AccuracyMetrics(y_true, y_pred_1, sample_duration,
                               sample_offset, threshold="max_f_score")
        am_2 = AccuracyMetrics(y_true, y_pred_2, sample_duration,
                               sample_offset, threshold="max_f_score")
        assert am_1.threshold_method == "max_f_score"
        assert am_1.threshold <= 0.9
        assert am_2.threshold <= 0.7

        assert am_1.sample_tp == am_2.sample_tp
        assert am_1.sample_tn == am_2.sample_tn
        assert am_1.sample_fp == am_2.sample_fp
        assert am_1.sample_fn == am_2.sample_fn

    def test_threshold_bad(self):
        """
        If a bad threshold value is given, a ValueError is raised.
        """
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
        with pytest.raises(ValueError):
            AccuracyMetrics(y_true, y_pred, sample_duration=1, sample_offset=1,
                            threshold="bad")
        with pytest.raises(ValueError):
            AccuracyMetrics(y_true, y_pred, sample_duration=1, sample_offset=1,
                            threshold=-1)

    def test_general_attributes(self):
        """
        Verify that the general attributes are correct.
        """
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

        sample_duration = 1
        sample_offset = 1

        am = AccuracyMetrics(y_true, y_pred, sample_duration,
                             sample_offset, threshold=0.5)

        assert am.sample_duration == pytest.approx(1.)
        assert am.sample_offset == pytest.approx(1.)
        assert am.n_true_seizures == 1
        assert am.total_duration == pytest.approx(1.0*60)
        assert am.n_samples == 60

    def test_sample_metrics(self):
        """
        Verify that sample-based metrics are correctly calculated.
        """
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

        sample_duration = 1
        sample_offset = 1

        am = AccuracyMetrics(y_true, y_pred, sample_duration,
                             sample_offset, threshold=0.5)

        assert .0 < am.sample_roc_auc < 1.
        assert .0 < am.sample_prc_auc < 1.
        assert am.sample_tp == 17
        assert am.sample_tn == 32
        assert am.sample_fp == 8
        assert am.sample_fn == 3
        assert am.sample_sensitivity == pytest.approx(0.85, abs=0.0001)
        assert am.sample_specificity == pytest.approx(0.8, abs=0.0001)
        assert am.sample_precision == pytest.approx(0.68, abs=0.0001)
        assert am.sample_recall == pytest.approx(0.85, abs=0.0001)
        assert am.sample_f_score == pytest.approx(0.7556, abs=0.0001)
        assert am.sample_accuracy == pytest.approx(0.8167, abs=0.0001)
        assert am.sample_balanced_accuracy == pytest.approx(0.8250, abs=0.0001)

    def test_event_conversion(self):
        """
        Verify that events are correctly extracted from lists of samples.
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        expected_events_true = [(5, 9),
                                (17, 26),
                                (29, 33),
                                (35, 39),
                                (43, 55),
                                (55, 59)]

        expected_events_true_extended = [(4, 11),
                                         (16, 28),
                                         (28, 35),
                                         (34, 41),
                                         (42, 57),
                                         (54, 61)]

        expected_events_pred = [(0, 5),
                                (10, 17),
                                (19, 28),
                                (41, 49),
                                (59, 63)]

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=12)

        assert am.event_minimum_overlap == pytest.approx(2.)
        assert am.event_preictal_tolerance == pytest.approx(1.)
        assert am.event_postictal_tolerance == pytest.approx(2.)
        assert am.event_minimum_separation == pytest.approx(2.)
        assert am.event_maximum_duration == pytest.approx(12.)

        for actual, expected in zip(am.events_true, expected_events_true):
            assert actual[0] == pytest.approx(expected[0])
            assert actual[1] == pytest.approx(expected[1])

        for actual, expected in zip(am.events_true_extended,
                                    expected_events_true_extended):
            assert actual[0] == pytest.approx(expected[0])
            assert actual[1] == pytest.approx(expected[1])

        for actual, expected in zip(am.events_pred, expected_events_pred):
            assert actual[0] == pytest.approx(expected[0])
            assert actual[1] == pytest.approx(expected[1])

    def test_event_conversion_one_big_event(self):
        """
        Verify that events are correctly extracted from lists of samples.
        """
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

        expected_events_true = [(0, 13)]

        expected_events_true_extended = [(-1, 15)]

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=24)

        for actual, expected in zip(am.events_true, expected_events_true):
            assert actual[0] == pytest.approx(expected[0])
            assert actual[1] == pytest.approx(expected[1])

        for actual, expected in zip(am.events_true_extended,
                                    expected_events_true_extended):
            assert actual[0] == pytest.approx(expected[0])
            assert actual[1] == pytest.approx(expected[1])

    def test_event_metrics(self):
        """
        Verify that events metrics are correct.
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=12)

        assert am.n_true_seizures == 6
        assert am.event_tp == 3
        assert am.event_fp == 2
        assert am.event_sensitivity == pytest.approx(0.5, abs=0.0001)
        assert am.event_precision == pytest.approx(0.6, abs=0.0001)
        assert am.event_recall == pytest.approx(0.5, abs=0.0001)
        assert am.event_f_score == pytest.approx(0.5455, abs=0.0001)
        assert am.event_false_detections_per_hour == pytest.approx(114.29, abs=0.01)
        assert am.event_false_detections_per_interictal_hour == pytest.approx(600, abs=0.01)
        assert am.event_average_detection_delay == pytest.approx(1.3333, abs=0.0001)

    def test_y_true_all_zeros(self):
        """
        Make sure that the classes handles the case where all true classes
        are zeros.
        """
        y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=12)

    def test_y_true_all_ones(self):
        """
        Make sure that the classes handles the case where all true classes
        are ones.
        """
        y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=12)

    def test_y_pred_all_zeros(self):
        """
        Make sure that the classes handles the case where all predicted classes
        are zeros.
        """
        y_true = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=12)

    def test_y_pred_all_ones(self):
        """
        Make sure that the classes handles the case where all predicted classes
        are ones.
        """
        y_true = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=12)

    def test_y_true_y_pred_all_zeros(self):
        """
        Make sure that the classes handles the case where all true and
        predicted values are zero (i.e. no true seizure, no predicted seizure).
        """
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=1,
                             sample_offset=1)

    def test_y_true_y_pred_all_ones(self):
        """
        Make sure that the classes handles the case where all true and
        predicted values are one (i.e. one true seizure, one predicted seizure).
        """
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=1,
                             sample_offset=1)

    def test_as_dict(self):
        """
        Verify that all non-iterable attributes are indeed saved in the dict
        given by the as_dict() method.
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=12)

        expected_dict = {"sample_duration": am.sample_duration,
                         "sample_offset": am.sample_offset,
                         "threshold_method": am.threshold_method,
                         "threshold": am.threshold,
                         "event_minimum_overlap": am.event_minimum_overlap,
                         "event_preictal_tolerance": am.event_preictal_tolerance,
                         "event_postictal_tolerance": am.event_postictal_tolerance,
                         "event_minimum_separation": am.event_minimum_separation,
                         "event_maximum_duration": am.event_maximum_duration,
                         "n_samples": am.n_samples,
                         "n_true_seizures": am.n_true_seizures,
                         "total_duration": am.total_duration,
                         "sample_roc_auc": am.sample_roc_auc,
                         "sample_prc_auc": am.sample_prc_auc,
                         "sample_tp": am.sample_tp,
                         "sample_tn": am.sample_tn,
                         "sample_fp": am.sample_fp,
                         "sample_fn": am.sample_fn,
                         "sample_sensitivity": am.sample_sensitivity,
                         "sample_specificity": am.sample_specificity,
                         "sample_precision": am.sample_precision,
                         "sample_recall": am.sample_recall,
                         "sample_f_score": am.sample_f_score,
                         "sample_accuracy": am.sample_accuracy,
                         "sample_balanced_accuracy": am.sample_balanced_accuracy,
                         "event_tp": am.event_tp,
                         "event_fp": am.event_fp,
                         "event_sensitivity": am.event_sensitivity,
                         "event_precision": am.event_precision,
                         "event_recall": am.event_recall,
                         "event_f_score": am.event_f_score,
                         "event_false_detections_per_hour": am.event_false_detections_per_hour,
                         "event_false_detections_per_interictal_hour": am.event_false_detections_per_interictal_hour,
                         "event_average_detection_delay": am.event_average_detection_delay
                         }

        assert am.as_dict() == expected_dict

    def test_pickle(self):
        """
        AccuracyMetrics should be pickleable.
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                           0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                           0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        am = AccuracyMetrics(y_true, y_pred,
                             sample_duration=4, sample_offset=1,
                             threshold=0.5, event_minimum_overlap=2,
                             event_preictal_tolerance=1,
                             event_postictal_tolerance=2,
                             event_minimum_separation=2,
                             event_maximum_duration=12)

        with tempfile.TemporaryFile() as tmpfile:
            pickle.dump(am, tmpfile)

    def test_non_flat_array(self):
        """
        No support for arrays with more than one dimensions., see issue #3.
        """
        y_true = np.array([[0], [0], [1], [1]])
        with pytest.raises(ValueError):
            am = AccuracyMetrics(y_true, y_true, sample_duration=4,
                                 sample_offset=2)

        y_true_flattened = y_true.flatten()
        am = AccuracyMetrics(y_true_flattened, y_true_flattened, 4, 2)
