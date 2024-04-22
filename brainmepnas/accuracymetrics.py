# -*- coding: utf-8 -*-

# import built-in module
import dataclasses
from typing import Union, List, Tuple

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
    threshold_method: str   # "fixed" or "max_f_score"
    threshold: float

    # Sample-based metrics
    sample_tp: int
    sample_tn: int
    sample_fp: int
    sample_fn: int
    sample_sensitivity: float
    sample_specificity: float
    sample_precision: float
    sample_recall: float    # = sensitivity
    sample_f_score: float

    # Event-based metrics
    events_true: List[Tuple[float, float]]
    events_true_extended: List[Tuple[float, float]]
    events_pred: List[Tuple[float, float]]
    event_tp: int
    event_fp: int
    event_sensitivity: float
    event_precision: float
    event_recall: float     # = sensitivity
    event_f_score: float
    event_false_detections_per_hour: float
    event_average_detection_delay: float

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 sample_duration: float, sample_offset: float,
                 threshold: Union[str, float] = 0.5,
                 event_minimum_overlap: float = None,
                 event_preictal_tolerance: float = 30,
                 event_postictal_tolerance: float = 60,
                 event_minimum_separation: float = 90,
                 event_maximum_duration: float = 360):
        """
        Calculate sample- and event-based accuracy metrics from arrays of
        predicted and true labels for each samples.

        The sample and event distinction is made in accordance with the
        proposed SzCORE - Seizure Community Open-source Research Evaluation
        framework[1]. Portions of the code were adapted from the implementation
        of the sz-validation-framework package, available on Github:
        https://github.com/esl-epfl/sz-validation-framework

        A future goal is to formally integrate the scoring module of
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
        :param sample_offset: duration between the start of two consecutive
        samples in seconds. For example, samples with a duration of 4 seconds
        and an offset of 1 second would have the following start and stop
        times:
            sample 0: 0s to 4s
            sample 1: 1s to 5s
            sample 2: 2s to 6s
            etc.
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
            self.threshold_method = "max_f_score"
        elif 0 <= float(threshold) <= 1:
            self.threshold = threshold
            self.threshold_method = "fixed"
        else:
            raise ValueError("threshold should be either a number between 0 "
                             "and 1 or one of 'max_f_score'.")

        y_true = np.where(y_true > 0.5, 1, 0)
        y_pred = np.where(y_pred > self.threshold, 1, 0)

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
        # Adapted from
        # https://github.com/esl-epfl/sz-validation-framework/blob/main/timescoring/scoring.py
        # We create a list of events, where each event is represented by a
        # (start, end) tuple, with the start and end values expressed in
        # seconds from the start of the provided data matrix (i.e. sample 0 in
        # y_pred starts at time 0 seconds).

        self.events_true = self._get_events(y_true, sample_duration,
                                            sample_offset,
                                            event_minimum_separation,
                                            event_maximum_duration)
        self.events_true_extended = self._extend_events(self.events_true,
                                                        event_preictal_tolerance,
                                                        event_postictal_tolerance)
        self.events_pred = self._get_events(y_pred, sample_duration,
                                            sample_offset,
                                            event_minimum_separation,
                                            event_maximum_duration)

        if event_minimum_overlap is None:
            event_minimum_overlap = 1e-6

        self.n_true_seizures = len(self.events_true)

        # True positive if a predicted event partially overlaps with a true
        # event.
        # Detection delay is the time between the true seizure start
        # (NOT extended) and the predicted detection. The delay is positive if
        # the prediction occurs after the true start, and negative if it occurs
        # before the true start (in the pre-ictal tolerance).
        self.event_tp = 0
        detection_delays = list()
        for true_event, true_event_extended in zip(self.events_true,
                                                   self.events_true_extended):
            for pred_event in self.events_pred:
                overlap = (min(true_event_extended[1], pred_event[1]) -
                           max(true_event_extended[0], pred_event[0]))
                if overlap >= event_minimum_overlap:
                    self.event_tp += 1
                    delay = pred_event[0] - true_event[0]
                    detection_delays.append(delay)
        self.event_average_detection_delay = np.average(detection_delays)

        # False positive if a predicted event does not overlap with a true
        # event.
        self.event_fp = 0
        for pred_event in self.events_pred:
            overlaps = list()
            for true_event, true_event_extended in zip(self.events_true,
                                                       self.events_true_extended):
                overlap = (min(true_event_extended[1], pred_event[1]) -
                           max(true_event_extended[0], pred_event[0]))
                overlaps.append(overlap)

            # No positive overlaps means no true event associated to this
            # predicted event.
            if len(overlaps) > 0:
                if max(overlaps) < event_minimum_overlap:
                    self.event_fp += 1

        # Assuming there is at least one true seizure in the data.
        if self.n_true_seizures > 0:
            self.event_sensitivity = self.event_tp / self.n_true_seizures
        else:
            self.event_sensitivity = np.nan
        self.event_recall = self.event_sensitivity

        if self.event_tp + self.event_fp == 0:
            self.event_precision = np.nan
            self.event_f_score = np.nan
        else:
            self.event_precision = self.event_tp / (self.event_tp + self.event_fp)
            self.event_f_score = (2 * (self.event_precision * self.event_recall) /
                                  (self.event_precision + self.event_recall))
        total_duration = (len(y_true) - 1) * sample_offset + sample_duration
        self.event_false_detections_per_hour = self.event_fp / total_duration * 3600

    def as_dict(self) -> dict:
        """
        Returns all non-iterable attributes in a dictionary with attribute
        names as keys. The dict can be used to save attributes to csv, for
        example.
        """
        d = {"n_true_seizures": self.n_true_seizures,
             "roc_auc": self.roc_auc,
             "prc_auc": self.prc_auc,
             "threshold_method": self.threshold_method,
             "threshold": self.threshold,
             "sample_tp": self.sample_tp,
             "sample_tn": self.sample_tn,
             "sample_fp": self.sample_fp,
             "sample_fn": self.sample_fn,
             "sample_sensitivity": self.sample_sensitivity,
             "sample_specificity": self.sample_specificity,
             "sample_precision": self.sample_precision,
             "sample_recall": self.sample_recall,
             "sample_f_score": self.sample_f_score,
             "event_tp": self.event_tp,
             "event_fp": self.event_fp,
             "event_sensitivity": self.event_sensitivity,
             "event_precision": self.event_precision,
             "event_recall": self.event_recall,
             "event_f_score": self.event_f_score,
             "event_false_detections_per_hour": self.event_false_detections_per_hour,
             "event_average_detection_delay": self.event_average_detection_delay
             }

        return d

    @staticmethod
    def _get_events(y: np.ndarray,
                    sample_duration: float, sample_offset: float,
                    event_minimum_separation: float,
                    event_maximum_duration: float) -> List[Tuple[float, float]]:
        """
        From a given binary array where 1 represent a sample with seizure
        detection, create a list of (start, end) tuples for every seizure
        event.

        The code is adapted from
        https://github.com/esl-epfl/sz-validation-framework.

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
        # Adapted from
        # https://github.com/esl-epfl/sz-validation-framework/blob/main/timescoring/annotations.py
        events = list()
        tmpEnd = []
        start_i = np.where(np.diff(np.array(y, dtype=int)) == 1)[0]
        end_i = np.where(np.diff(np.array(y, dtype=int)) == -1)[0]

        # No transitions and first sample is positive -> event is duration of
        # file
        if len(start_i) == 0 and len(end_i) == 0 and y[0]:
            events.append((0,
                           ((len(y) - 1) * sample_offset) + sample_duration))
        else:
            # Edge effect - First sample is an event
            if y[0]:
                events.append(
                    (0, (end_i[0] * sample_offset) + sample_duration))
                end_i = np.delete(end_i, 0)

            # Edge effect - Last event runs until end of file
            if y[-1]:
                if len(start_i):
                    tmpEnd = [((start_i[-1] + 1) * sample_offset,
                               ((len(y) - 1) * sample_offset) + sample_duration)]
                    start_i = np.delete(start_i, len(start_i) - 1)

            # Add all events
            start_i += 1
            end_i += 1
            for i in range(len(start_i)):
                events.append((start_i[i] * sample_offset,
                               ((end_i[i] - 1) * sample_offset) + sample_duration))
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
        shorter_events = merged_events.copy()

        for i, event in enumerate(shorter_events):
            if event[1] - event[0] > event_maximum_duration:
                shorter_events[i] = (event[0],
                                     event[0] + event_maximum_duration)
                shorter_events.insert(i + 1,
                                      (event[0] + event_maximum_duration,
                                       event[1]))

        return shorter_events

    @staticmethod
    def _extend_events(events: List[tuple], preictal: float,
                       postictal: float) -> List[Tuple[float, float]]:
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
