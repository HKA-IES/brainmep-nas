# -*- coding: utf-8 -*-

# import built-in module

# import third-party modules
import pytest
import numpy as np

# import your own module
from brainmepnas.metrics import (AccuracyMetrics, confusion_matrix,
                                 detected_seizures, false_detections_per_hour,
                                 average_detection_delay)


class TestMetrics:
    """
    The calculation of metrics should be very robust in order to avoid wrong
    results and analyzes/conclusions down the line.

    The metrics.py module is expected to achieve 100% coverage and redundant
    examples are desired.

    The tests are worked out by hand in test_metrics.ods.
    """

    # Tests for self-written methods
    def test_confusion_matrix(self):
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

        tp_expected = 17
        tn_expected = 32
        fp_expected = 8
        fn_expected = 3

        tp, tn, fp, fn = confusion_matrix(y_true, y_pred, 0.5, 0.5)

        assert tp == tp_expected
        assert tn == tn_expected
        assert fp == fp_expected
        assert fn == fn_expected

    def test_detected_seizures_one_true_seizure_one_detection(self):
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

        seizures_expected = 1
        nb_detected_seizures_expected = 1

        nb_detected_seizures, seizures = detected_seizures(y_true, y_pred,
                                                           0.5, 0.5)

        assert nb_detected_seizures == nb_detected_seizures_expected
        assert seizures == seizures_expected

    def test_detected_seizures_two_true_seizures_one_detection(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                           1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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

        seizures_expected = 2
        nb_detected_seizures_expected = 1

        nb_detected_seizures, seizures = detected_seizures(y_true, y_pred,
                                                           0.5, 0.5)

        assert nb_detected_seizures == nb_detected_seizures_expected
        assert seizures == seizures_expected

    def test_detected_seizures_two_true_seizures_two_detections(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                           1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1, 0, 1, 0, 0])

        seizures_expected = 2
        nb_detected_seizures_expected = 2

        nb_detected_seizures, seizures = detected_seizures(y_true, y_pred,
                                                           0.5, 0.5)

        assert nb_detected_seizures == nb_detected_seizures_expected
        assert seizures == seizures_expected

    def test_false_detections_per_hour_short_detection_window_no_overlap(self):
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

        window_length = 1
        window_overlap = 0
        detection_window_length = 4
        fdh_expected = 180

        fdh = false_detections_per_hour(y_true, y_pred, 0.5, 0.5,
                                        window_length, window_overlap,
                                        detection_window_length)

        assert fdh == fdh_expected

    def test_false_detections_per_hour_long_detection_window_no_overlap(self):
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

        window_length = 1
        window_overlap = 0
        detection_window_length = 20
        fdh_expected = 60

        fdh = false_detections_per_hour(y_true, y_pred, 0.5, 0.5,
                                        window_length, window_overlap,
                                        detection_window_length)

        assert fdh == fdh_expected

    def test_false_detections_per_hour_short_detection_window_with_overlap(self):
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

        window_length = 1
        window_overlap = 0.5
        detection_window_length = 3
        fdh_expected = 354.0983607

        fdh = false_detections_per_hour(y_true, y_pred, 0.5, 0.5,
                                        window_length, window_overlap,
                                        detection_window_length)

        assert fdh == pytest.approx(fdh_expected)

    def test_false_detections_per_hour_long_detection_window_with_overlap(self):
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

        window_length = 1
        window_overlap = 0.5
        detection_window_length = 10
        fdh_expected = 118.0327869

        fdh = false_detections_per_hour(y_true, y_pred, 0.5, 0.5,
                                        window_length, window_overlap,
                                        detection_window_length)

        assert fdh == pytest.approx(fdh_expected)

    def test_average_detection_delay_one_true_seizure_no_overlap(self):
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

        window_length = 1
        window_overlap = 0
        avg_delay_expected = 2

        avg_delay = average_detection_delay(y_true, y_pred, 0.5, 0.5,
                                            window_length, window_overlap)

        assert avg_delay == pytest.approx(avg_delay_expected)

    def test_average_detection_delay_one_true_seizure_with_overlap(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 1, 0, 1, 0, 0])

        window_length = 1
        window_overlap = 0.5
        avg_delay_expected = 0.5

        avg_delay = average_detection_delay(y_true, y_pred, 0.5, 0.5,
                                            window_length, window_overlap)

        assert avg_delay == pytest.approx(avg_delay_expected)

    def test_average_detection_delay_two_true_seizures_no_overlap(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                           1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 1, 1, 1, 0, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
                           1, 0, 0, 0, 0, 1, 0, 1, 0, 0])

        window_length = 1
        window_overlap = 0
        avg_delay_expected = 1

        avg_delay = average_detection_delay(y_true, y_pred, 0.5, 0.5,
                                            window_length, window_overlap)

        assert avg_delay == pytest.approx(avg_delay_expected)

    def test_average_detection_delay_two_true_seizures_with_overlap(self):
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                           1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 0, 1, 0, 0, 0, 0, 0, 1,
                           1, 0, 0, 0, 0, 1, 0, 1, 0, 0])

        window_length = 1
        window_overlap = 0.5
        avg_delay_expected = 1

        avg_delay = average_detection_delay(y_true, y_pred, 0.5, 0.5,
                                            window_length, window_overlap)

        assert avg_delay == pytest.approx(avg_delay_expected)

    # Test for dataclasses
    def test_accuracy_metrics_fixed_threshold(self):
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
        nb_elements = len(y_true)
        window_length = 1
        window_overlap = 0
        detection_window_length = 20
        am = AccuracyMetrics(y_true, y_pred, window_length, window_overlap,
                             detection_window_length, threshold=0.5)

        assert 0 < am.roc_auc < 1
        assert 0 < am.prc_auc < 1
        assert 0 < am.sample_f_score < 1
        assert am.threshold == pytest.approx(0.5)
        assert (am.sample_tp + am.sample_tn + am.sample_fp +
                am.false_negatives) == nb_elements
        assert 0 < am.sample_sensitivity < 1
        assert 0 < am.sample_specificity < 1
        assert 0 < am.sample_precision < 1
        assert am.sample_recall == pytest.approx(am.sample_sensitivity)
        assert am.seizures_true == 1
        assert am.seizures_detected == 1
        assert am.false_detections_per_hour == pytest.approx(60)
        assert am.average_detection_delay == pytest.approx(2)

    def test_accuracy_metrics_threshold_max_f_score(self):
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
        nb_elements = len(y_true)
        window_length = 1
        window_overlap = 0
        detection_window_length = 20
        am = AccuracyMetrics(y_true, y_pred, window_length, window_overlap,
                             detection_window_length, threshold="max_f_score")

        assert 0 < am.roc_auc < 1
        assert 0 < am.prc_auc < 1
        assert 0 < am.sample_f_score < 1
        assert am.threshold != pytest.approx(0.5)
        assert (am.sample_tp + am.sample_tn + am.sample_fp +
                am.false_negatives) == nb_elements
        assert 0 < am.sample_sensitivity < 1
        assert 0 < am.sample_specificity < 1
        assert 0 < am.sample_precision < 1
        assert am.sample_recall == pytest.approx(am.sample_sensitivity)
        assert am.seizures_true == 1
        assert am.seizures_detected == 1
        assert am.false_detections_per_hour == pytest.approx(60)
        assert am.average_detection_delay == pytest.approx(2)
