# -*- coding: utf-8 -*-

# import built-in module
import dataclasses
from typing import Union, List

# import third-party modules
import sklearn.metrics as sk_metrics
import numpy as np

# import your own module


# TODO: Consider integrating sz-validation-framework once package is stable and
#  available.

@dataclasses.dataclass
class AccuracyMetrics:
    """
    Compute and store accuracy metrics from arrays of true and predicted
    labels.
    """

    n_true_seizures: int
    roc_auc: float
    prc_auc: float
    threshold: float

    # Sample-based metrics
    sample_tp: int
    sample_tn: int
    sample_fp: int
    sample_fn: int
    sample_sensitivity: float
    sample_specificity: float
    sample_precision: float
    sample_recall: float
    sample_f_score: float

    # Event-based metrics
    event_tp: int
    event_tn: int
    event_fp: int
    event_fn: int
    event_sensitivity: float
    event_specificity: float
    event_precision: float
    event_recall: float
    event_f_score: float
    event_false_detections_per_hour: float
    event_average_detection_delay: float

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 threshold: Union[str, float],
                 window_length, window_overlap, detection_window_length):
        """
        Calculate sample- and event-based accuracy metrics from arrays of
        predicted and true labels for each samples.

        The sample and event distinction is made in accordance with the
        proposed SzCORE - Seizure Community Open-source Research Evaluation
        framework[1]. Portions of the code were adapted from the implementation
        of the sz-validation-framework package, available on Github:
        https://github.com/esl-epfl/sz-validation-framework

        A future goal is to eventually integrate the scoring module of
        sz-validation-framework in this class.

        Reference:
        [1] J. Dan et al., “SzCORE: A Seizure Community Open-source Research
        Evaluation framework for the validation of EEG-based automated seizure
        detection algorithms.” arXiv, Feb. 23, 2024. Accessed: Feb. 27, 2024.
        [Online]. Available: http://arxiv.org/abs/2402.13005

        :param y_true: array of true labels. Expected values are either 0
        (no seizure) or (seizure).
        :param y_pred: array of predicted labels. Expected values between 0 and
        1.
        :param threshold: either a fixed threshold or one of the following:
            "max_f_score": threshold which maximizes the f score.
        :param sample_duration: duration of a sample (label) in seconds.
        :param sample_overlap: overlap duration for samples (labels) in
        seconds.
        :param event_minimum_overlap: minimum overlap between predicted and
        true events for a detection, in seconds. Default is any overlap (as in
        [1]).
        :param event_preictal_tolerance: A predicted seizure is counted as a
        true prediction if it is predicted up to event_preictal_tolerance
        seconds before a true seizure. Default is 30 seconds (as in [1]).
        :param event_postictal_tolerance: A predicted seizure is counted as a
        true prediction if it is predicted up to event_postictal_tolerance
        seconds after a true seizure. Default is 60 seconds (as in [1]).
        :param event_minimum_separation: Events that are separated by less than
        event_minimum_separation seconds are merged. Default is 90 seconds
        (combined pre- and post-ictal tolerance, as in [1]).
        :param event_maximum_duration: Events that are longer than
        event_maximum_duration seconds are split in events with the maximum
        duration. This is done after the merging of close events (see
        event_minimum_separation). Default is 300 seconds (as in [1]).
        """
        # ROC curve
        roc_fpr, roc_tpr, _ = sk_metrics.roc_curve(y_true, y_pred)
        self.roc_auc = sk_metrics.auc(roc_fpr, roc_tpr)

        # Precision-recall curve
        (prc_precision,
         prc_recall,
         prc_thresholds) = sk_metrics.precision_recall_curve(y_true, y_pred)
        self.prc_auc = sk_metrics.auc(prc_recall, prc_precision)
        f_scores = 2 * (prc_precision * prc_recall) / (
                prc_precision + prc_recall)
        nan_idx = np.argwhere(np.isnan(f_scores))
        f_scores = np.delete(f_scores, nan_idx)
        prc_thresholds = np.delete(prc_thresholds, nan_idx)
        self.sample_f_score = np.max(f_scores)

        if threshold == "max_f_score":
            self.threshold = float(prc_thresholds[np.argmax(f_scores)])
        elif 0 < threshold < 1:
            self.threshold = threshold
        else:
            raise ValueError("threshold should be either a number between 0 "
                             "and 1 or one of 'max_f_score'.")

        y_true = np.where(y_true > 0.5, 1, 0)
        y_pred = np.where(y_pred > threshold, 1, 0)

        # Sample-based metrics
        (self.sample_tn,
         self.sample_fp,
         self.sample_fn,
         self.sample_tp) = sk_metrics.confusion_matrix(y_true, y_pred).ravel()

        self.sample_sensitivity = (self.sample_tp /
                                   (self.sample_tp + self.sample_fn))
        self.sample_specificity = (self.sample_tn /
                                   (self.sample_tn + self.sample_fp))
        self.sample_precision = (self.sample_tp /
                                 (self.sample_tp + self.sample_fp))
        self.sample_recall = self.sample_sensitivity

        # Event-based metrics
        # Adapted from https://github.com/esl-epfl/sz-validation-framework/blob/main/timescoring/scoring.py
        # We create a list of events, where each event is represented by a
        # (start, end) tuple, with the start and end values expressed in
        # seconds from the start of the provided data matrix (i.e. sample 0 in
        # y_pred starts at time 0 seconds).

        # TODO: You are here ...

        # Nb detected seizures
        (self.seizures_detected,
         self.seizures_true) = detected_seizures(y_true, y_pred,
                                                 0.5, self.threshold)

        # False detections per hour
        self.false_detections_per_hour \
            = false_detections_per_hour(y_true, y_pred, 0.5, self.threshold,
                                        window_length,
                                        window_overlap,
                                        detection_window_length)

        # Detection delay
        self.average_detection_delay = average_detection_delay(y_true, y_pred,
                                                               0.5,
                                                               self.threshold,
                                                               window_length,
                                                               window_overlap)


@dataclasses.dataclass
class HardwareMetrics:
    """
    Store hardware metrics.
    """

    energy: float
    time: float
    ram_memory: int
    flash_memory: int


# TODO: Consider taking out the thresholding process from the calculations
def confusion_matrix(y_true, y_pred, threshold_true, threshold_pred) -> tuple[int, int, int, int]:
    """
    Calculate the true positives, true negatives, false positives, and false
    negatives.
    :param y_true: array of true values.
    :param y_pred: array of predicted values.
    :param threshold_true: threshold for true values.
    :param threshold_pred: threshold for predicted values.
    :return: (true positives, true negatives, false positives, false negatives)
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for p, t in zip(y_pred, y_true):
        if t >= threshold_true and p >= threshold_pred:
            tp += 1
        elif t >= threshold_true:
            fn += 1
        elif t < threshold_true and p >= threshold_pred:
            fp += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def detected_seizures(y_true, y_pred, threshold_true, threshold_pred) -> tuple[int, int]:
    """
    A seizure is considered detected if there is at least one predicted seizure
    in the duration of a continuous true seizure.

    This gives insights into whether we have missed any seizure.

    :param y_true: array of true values.
    :param y_pred: array of predicted values.
    :param threshold_true: threshold for true values.
    :param threshold_pred: threshold for predicted values.
    :return: (detected seizures, true seizures)
    """
    true_seizure_ongoing = False
    nb_true_seizures = 0

    detected_seizure_ongoing = False
    nb_detected_seizures = 0

    for p, t in zip(y_pred, y_true):
        if t > threshold_true:
            if not true_seizure_ongoing:
                true_seizure_ongoing = True
                nb_true_seizures += 1
        else:
            true_seizure_ongoing = False
            detected_seizure_ongoing = False

        if p >= threshold_pred and t > threshold_true:
            if not detected_seizure_ongoing:
                detected_seizure_ongoing = True
                nb_detected_seizures += 1

    return nb_detected_seizures, nb_true_seizures


def false_detections_per_hour(y_true, y_pred, threshold_true, threshold_pred,
                              window_length, window_overlap,
                              detection_window_length) -> float:
    """
    We assume that each detection is followed by a stimulation of duration
    detection_window_length. If two distinct samples are false positives within
    that window, they will be counted as a single detection.

    Everytime y_pred detects a seizure, a window of detection_window_length is
    established. If y_true is positive within this window, this counts as
    a good detection (seizure correctly detected). If y_true is never positive
    within this window, this counts as a bad detection (false positive).
    """
    fd = 0

    detection_window_remaining_time = 0
    true_seizure_in_window = False
    detection_window = False

    delta_time = window_length - window_overlap

    for p, t in zip(y_pred, y_true):
        if not detection_window:
            if p >= threshold_pred:
                detection_window_remaining_time = detection_window_length
                true_seizure_in_window = False
                detection_window = True
        else:
            if t >= threshold_true:
                true_seizure_in_window = True

            if detection_window_remaining_time <= 0:
                if not true_seizure_in_window:
                    fd += 1
                true_seizure_in_window = False
                detection_window = False
                detection_window_remaining_time = 0

        if detection_window:
            detection_window_remaining_time -= delta_time

    if detection_window and not true_seizure_in_window:
        fd += 1

    total_length = (window_length +
                    (len(y_true)-1)*(window_length-window_overlap))
    fdh = fd / total_length * 3600
    return fdh


def average_detection_delay(y_true, y_pred, threshold_true, threshold_pred,
                            window_length, window_overlap) -> float:
    """
    Detection delay is time between seizure start and first detection.
    Can be positive or negative.
    """
    true_seizure_ongoing = False
    nb_true_seizures = 0

    detected_seizure_ongoing = False
    nb_detected_seizures = 0
    time = 0

    detection_delay_values = []

    delta_time = window_length - window_overlap

    for p, t in zip(y_pred, y_true):
        if t > threshold_true:
            if not true_seizure_ongoing:
                true_seizure_ongoing = True
                nb_true_seizures += 1
                seizure_time = time
        else:
            true_seizure_ongoing = False
            detected_seizure_ongoing = False

        if p >= threshold_pred and t > threshold_true:
            if not detected_seizure_ongoing:
                detected_seizure_ongoing = True
                nb_detected_seizures += 1
                detection_delay_values.append(time - seizure_time)

        time += delta_time

    avg_detection_delay = np.mean(detection_delay_values)

    return float(avg_detection_delay)

# TODO: Add tests for get_events
def get_events(y: np.ndarray, sample_duration: float, sample_overlap: float,
               event_minimum_separation: float, event_maximum_duration: float) -> List[tuple]:
    """
    From a given binary array where 1 represent a sample with seizure
    detection, create a list of (start, end) tuples for every seizure event.

    The code is adapted from https://github.com/esl-epfl/sz-validation-framework.

    :param y: array of seizure labels. Expected values are either 0
    (no seizure) or 1 (seizure).
    :param sample_duration: duration of a sample (label) in seconds.
    :param sample_overlap: overlap duration for samples (labels) in
    seconds.
    :param event_minimum_separation: Events that are separated by less than
    event_minimum_separation seconds are merged.
    :param event_maximum_duration: Events that are longer than
    event_maximum_duration seconds are split in events with the maximum
    duration. This is done after the merging of close events (see
    event_minimum_separation).
    """
    # Adapted from https://github.com/esl-epfl/sz-validation-framework/blob/main/timescoring/annotations.py
    samples_offset = sample_duration - sample_overlap
    events = list()
    tmpEnd = []
    start_i = np.where(np.diff(np.array(y, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(y, dtype=int)) == -1)[0]

    # No transitions and first sample is positive -> event is duration of file
    if len(start_i) == 0 and len(end_i) == 0 and y[0]:
        events.append((0, (len(y) * samples_offset) + sample_duration))
    else:
        # Edge effect - First sample is an event
        if y[0]:
            events.append((0, (end_i[0] * samples_offset) + sample_duration))
            end_i = np.delete(end_i, 0)
        # Edge effect - Last event runs until end of file
        if y[-1]:
            if len(start_i):
                tmpEnd = [(start_i[-1] * samples_offset,
                           (len(y) * samples_offset) + sample_duration)]
                start_i = np.delete(start_i, len(start_i) - 1)
        # Add all events
        start_i += 1
        end_i += 1
        for i in range(len(start_i)):
            events.append((start_i[i] * samples_offset, (end_i[i] * samples_offset) + sample_duration))
        events += tmpEnd  # add potential end edge effect

    # Merge close events
    merged_events = events.copy()
    i = 1
    while i < len(merged_events):
        event = merged_events[i]
        if event[0] - merged_events[i - 1][1] < event_minimum_separation:
            merged_events[i - 1] = (merged_events[i - 1][0], event[1])
            del merged_events[i]
            i -= 1
        i += 1

    # Split long events
    shorter_events = events.copy()

    for i, event in enumerate(shorter_events):
        if event[1] - event[0] > event_maximum_duration:
            shorter_events[i] = (event[0], event[0] + event_maximum_duration)
            shorter_events.insert(i + 1,
                                 (event[0] + event_maximum_duration, event[1]))

    return shorter_events


# TODO: Add tests for extend_events
def extend_events(events: list, preictal: float, postictal: float) -> List[tuple]:
    """
    Extend events in the pre- and post-ictal directions.

    The code is adapted from https://github.com/esl-epfl/sz-validation-framework.

    :param events: list of events.
    :param preictal: Time to extend before each event, in seconds.
    :param postictal: Time to extend after each event, in seconds.
    :return: extended_events: list of extended events.
    """
    extended_events = events.copy()

    for i, event in enumerate(extended_events):
        extended_events[i] = (event[0] - preictal, event[1] + postictal)

    return extended_events
