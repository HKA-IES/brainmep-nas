# -*- coding: utf-8 -*-

# import built-in module
import dataclasses
from typing import Union, List, Tuple, Literal
import numbers
import warnings

# import third-party modules
import sklearn.metrics as sk_metrics
import numpy as np
from timescoring.annotations import Annotation
from timescoring.scoring import SampleScoring, EventScoring

# import your own module


@dataclasses.dataclass
class AccuracyMetrics:
    """
    Compute and store accuracy metrics from arrays of true and predicted
    labels.

    Sample-based metrics are calculated on a sample-by-sample basis. These
    metrics are ubiquitous in machine learning.

    Event-based metrics represent more adequately the detection of seizure
    events, whereas a seizure is assumed to last for many samples.

    The sample and event distinction is made in accordance with the proposed
    SzCORE - Seizure Community Open-source Research Evaluation framework[1].
    Portions of the code were adapted from the implementation of the
    sz-validation-framework package, available on Github:
    https://github.com/esl-epfl/sz-validation-framework

    A future goal is to formally integrate the scoring module of
    sz-validation-framework in this class.

    References
    ----------
    [1] J. Dan et al., “SzCORE: A Seizure Community Open-source Research
    Evaluation framework for the validation of EEG-based automated seizure
    detection algorithms.” arXiv, Feb. 23, 2024. Accessed: Feb. 27, 2024.
    [Online]. Available: http://arxiv.org/abs/2402.13005

    Attributes
    ----------
    sample_duration: float
        Duration of a single sample, in seconds.
    sample_offset: float
        Time offset between the beginning of a sample and the beginning of the
        following sample, in seconds.
    threshold_method: str
        Method used to set the threshold. Can be one of the following:
        - "fixed": threshold fixed by the user.
        - "max_f_score": threshold set to maximize the F-score.
    threshold: float
        Threshold to separate seizure from non-seizure at a sample level. A
        predicted value below the threshold corresponds to a non-seizure,
        whereas a predicted above or equal to the threshold corresponds to a
        seizure.
    event_minimum_rel_overlap : float
        Minimum relative overlap between predicted and true events for a
        detection, between 0 and 1. 0 indicates that any overlap is
        considered to be a proper detection, whereas 1 indicates that the
        predicted event should fully overlap the true event.
    event_preictal_tolerance : float
        A predicted seizure is counted as a true prediction if it is
        predicted up to event_preictal_tolerance seconds before a true
        seizure.
    event_postictal_tolerance : float
        A predicted seizure is counted as a true prediction if it is
        predicted up to event_postictal_tolerance seconds after a true
        seizure.
    event_minimum_separation : float
        Events that are separated by less than event_minimum_separation
        seconds are merged.
    event_maximum_duration : float
        Events that are longer than event_maximum_duration seconds are
        split in events with the maximum duration. This is done after the
        merging of close events (see event_minimum_separation).
    y_true: np.ndarray
        Array of true labels, where 0 corresponds to an interictal sample (no
        seizure) and 1 corresponds to an ictal sample (seizure).
    n_samples: int
        Number of samples.
    n_true_seizures: int
        Number of true seizure events in y_true.
    total_duration: float
        Total duration of y_true.
    y_pred: np.ndarray
        Array of predicted values (between 0 and 1).
    y_pred_post_threshold: np.ndarray
        Array of predicted labels. The threshold is applied to y_pred such that
        values < threshold are assigned the label 0 (interictal, no seizure)
        and values >= threshold are assigned the label 1 (ictal, seizure).
    sample_roc_auc: float
        Sample-based metric: Area under the receiver-operating curve. The 
        x-axis is the false positive rate (FPR = 1 - TNR) and the y-axis is the
        sensitivity (TPR).
    sample_prc_auc: float
        Sample-based metric: Area under the precision-recall curve.
    sample_tp: int
        Sample-based metric: true positives.
    sample_tn: int
        Sample-based metric: true negatives.
    sample_fp: int
        Sample-based metric: false positives.
    sample_fn: int
        Sample-based metric: false negatives.
    sample_sensitivity: float
        Sample-based metric: sensitivity (true positive rate).
            TPR = TP/(TP+FN)
    sample_specificity: float
        Sample-based metric: specificity (true negative rate).
            TNR = TN/(TN+FP)
    sample_precision: float
        Sample-based metric: precision (positive predictive value).
            PPV = TP/(TP+FP)
    sample_recall: float
        Sample-based metric: recall (equivalent to sensitivity, true positive
        rate).
            TPR = TP/(TP+FN)
    sample_f_score: float
        Sample-based metric: F-score (harmonic mean of precision and recall)
            F1 = 2*(PPV*TPR)/(PPV+TPR)
    sample_accuracy: float
        Sample-based metric: proportion of correct predictions over all
        predictions.
            ACC = (TP+TN)/(TP+TN+FP+FN)
    sample_balanced_accuracy: float
        Sample-based metric: arithmetic mean of sensitivity and specificity.
            BALACC = (TPR+TNR)/2
    events_true: List[Tuple[float, float]]
        List of true events, where each event is represented of a tuple of
        (start_time, end_time) for the event, in seconds from the beginning of
        y_true.
        Events are obtained by compiling a list of the start and end
        times of consecutive 1 labels in y_true.
    events_true_merged_split: List[Tuple[float, float]]
        List of true events, where each event is represented of a tuple of
        (start_time, end_time) for the event, in seconds from the beginning of
        y_true.
        Events are obtained by first compiling a list of the start and end
        times of consecutive 1 labels in y_true. Then, events that are
        separated by less than event_minimum_separation seconds are merged.
        Finally, events that are longer than event_maximum_duration seconds are
        split in events with the maximum duration.
    events_pred: List[Tuple[float, float]]
        List of predicted events, where each event is represented of a tuple of
        (start_time, end_time) for the event, in seconds from the beginning of
        y_true.
        Events are obtained by compiling a list of the start and end
        times of consecutive 1 labels in y_pred.
    events_pred_merged_split: List[Tuple[float, float]]
        List of predicted events, where each event is represented of a tuple of
        (start_time, end_time) for the event, in seconds from the beginning of
        y_true.
        Events are obtained by first compiling a list of the start and end
        times of consecutive 1 labels in y_pred. Then, events that are
        separated by less than event_minimum_separation seconds are merged.
        Finally, events that are longer than event_maximum_duration seconds are
        split in events with the maximum duration.
    event_tp: int
        Event-based metric: true positives.
    event_fp: int
        Event-based metric: false positives.
    event_sensitivity: float
        Event-based metric: sensitivity (true positive rate).
            TPR = TP/P
    event_precision: float
        Event-based metric: precision (positive predictive value).
            PPV = TP/(TP+FP)
    event_recall: float
        Event-based metric: recall (equivalent to sensitivity, true positive
        rate).
            TPR = TP/P
    event_f_score: float
        Event-based metric: F-score (harmonic mean of precision and recall)
            F1 = 2*(PPV*TPR)/(PPV+TPR)
    event_false_detections_per_hour: float
        Number of FP per recording hour.
    event_false_detections_per_interictal_hour: float
        Number of FP per interictal hour, where the total interictal time
        corresponds to the total duration of true seizure events.
    event_average_detection_delay: float
        Average delay between the beginning of a predicted event and the
        associated true event (without considering the pre-ictal tolerance).
        The delay is positive if the prediction occurs after the true start,
        and negative if it occurs before the true start (in the pre-ictal
        tolerance).

    # TODO: Check for correct values in edge cases (i.e. TP or TN = 0, ...)
    """
    sample_duration: float
    sample_offset: float
    threshold_method: Literal["fixed", "max_f_score"]
    threshold: float
    event_minimum_rel_overlap: float
    event_preictal_tolerance: float
    event_postictal_tolerance: float
    event_minimum_separation: float
    event_maximum_duration: float

    y_true: np.ndarray
    n_samples: int
    n_true_seizures: int
    total_duration: float

    y_pred: np.ndarray
    y_pred_post_threshold: np.ndarray

    # Sample-based metrics
    sample_roc_auc: float
    sample_prc_auc: float
    sample_tp: int
    sample_tn: int
    sample_fp: int
    sample_fn: int
    sample_sensitivity: float
    sample_specificity: float
    sample_precision: float
    sample_recall: float
    sample_f_score: float
    sample_accuracy: float
    sample_balanced_accuracy: float

    # Event-based metrics
    events_true: List[Tuple[float, float]]
    events_pred: List[Tuple[float, float]]
    events_true_merged_split: List[Tuple[float, float]]
    events_pred_merged_split: List[Tuple[float, float]]
    event_tp: int
    event_fp: int
    event_sensitivity: float
    event_precision: float
    event_recall: float
    event_f_score: float
    event_false_detections_per_hour: float
    event_false_detections_per_interictal_hour: float
    event_average_detection_delay: float

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 sample_duration: float, sample_offset: float,
                 threshold: Union[Literal["max_f_score"], float] = 0.5,
                 event_minimum_rel_overlap: float = 0,
                 event_preictal_tolerance: float = 30,
                 event_postictal_tolerance: float = 60,
                 event_minimum_separation: float = 90,
                 event_maximum_duration: float = 360):
        """
        Calculate sample- and event-based accuracy metrics from arrays of
        predicted and true labels for each sample.

        The sample and event distinction is made in accordance with the
        proposed SzCORE - Seizure Community Open-source Research Evaluation
        framework[1]. Portions of the code were adapted from the implementation
        of the sz-validation-framework package, available on Github:
        https://github.com/esl-epfl/sz-validation-framework

        A future goal is to formally integrate the scoring module of
        sz-validation-framework in this class.

        References
        ----------
        [1] J. Dan et al., “SzCORE: A Seizure Community Open-source Research
        Evaluation framework for the validation of EEG-based automated seizure
        detection algorithms.” arXiv, Feb. 23, 2024. Accessed: Feb. 27, 2024.
        [Online]. Available: http://arxiv.org/abs/2402.13005

        Parameters                                                                                                  
        ----------
        y_true : np.ndarray(int)
            1D array of true labels. Expected values are either 0 (no seizure)
            or 1 (seizure).
        y_pred : np.ndarray(float)
            1D array of predicted labels. Expected values between 0 and 1.
        threshold : str or float
            The threshold to apply. Either a fixed (float) threshold or 
            "max_f_score": for threshold which maximizes the f score.
        sample_duration : float
            Duration of a sample (signal window) in seconds.
        sample_offset : float
            Duration between the start of two consecutive
            samples in seconds. For example, samples with a duration of 4
            seconds and a stride of 1 second would have the following start and
            stop´times:
                sample 0: 0s to 4s
                sample 1: 1s to 5s
                sample 2: 2s to 6s
                etc.
        event_minimum_rel_overlap : float
            Minimum relative overlap between predicted and true events for a
            detection, between 0 and 1. 0 indicates that any overlap is
            considered to be a proper detection, whereas 1 indicates that the
            predicted event should fully overlap the true event.
            Default is 0 (any overlap, as in [1]).
        event_preictal_tolerance : float
            A predicted seizure is counted as a true prediction if it is
            predicted up to event_preictal_tolerance seconds before a true
            seizure. Default is 30 seconds (as in [1]).
        event_postictal_tolerance : float                                                                         
            A predicted seizure is counted as a true prediction if it is
            predicted up to event_postictal_tolerance seconds after a true
            seizure. Default is 60 seconds (as in [1]).
        event_minimum_separation : float                                                                           
            Events that are separated by less than event_minimum_separation
            seconds are merged. Default is 90 seconds (combined pre- and
            post-ictal tolerance, as in [1]).
        event_maximum_duration : float 
            Events that are longer than event_maximum_duration seconds are
            split in events with the maximum duration. This is done after the
            merging of close events (see event_minimum_separation).
            Default is 300 seconds (as in [1]).
        """
        # Check for validity of arguments
        if not ((y_true == 0) | (y_true == 1)).all():
            raise ValueError("y_true must be a binary array!")
        if (y_true != 1).all():
            warnings.warn("There are no seizures events in y_true. Is this "
                          "expected?")
        if y_true.ndim > 1:
            raise ValueError("y_true contains more than one dimension. Flatten"
                             "it to 1D before passing it to AccuracyMetrics.")
        self.y_true = y_true

        if not ((y_pred >= 0) & (y_pred <= 1)).all():
            raise ValueError("y_pred must contain values between 0 and 1.")
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred should have the same length.")
        if y_pred.ndim > 1:
            raise ValueError("y_pred contains more than one dimension. Flatten"
                             "it to 1D before passing it to AccuracyMetrics.")
        self.y_pred = y_pred

        if not sample_duration > 0:
            raise ValueError("sample_duration must be > 0 seconds.")
        self.sample_duration = sample_duration

        if not sample_offset > 0:
            raise ValueError("sample_offset must be > 0 seconds. For "
                             "non-overlapping samples: "
                             "sample_offset >= sample_duration")
        self.sample_offset = sample_offset

        if not 1 >= event_minimum_rel_overlap >= 0:
            raise ValueError("event_minimum_rel_overlap must be between 0 and"
                             " 1 (inclusive).")
        self.event_minimum_rel_overlap = event_minimum_rel_overlap

        if not event_preictal_tolerance >= 0:
            raise ValueError("event_preictal_tolerance must be >= 0 seconds.")
        self.event_preictal_tolerance = event_preictal_tolerance

        if not event_postictal_tolerance >= 0:
            raise ValueError("event_postictal_tolerance must be >= 0 seconds.")
        self.event_postictal_tolerance = event_postictal_tolerance

        if not event_minimum_separation >= 0:
            raise ValueError("event_minimum_separation must be >= 0 seconds.")
        self.event_minimum_separation = event_minimum_separation

        if not event_maximum_duration > 0:
            raise ValueError("event_maximum_duration must be > 0 seconds.")
        self.event_maximum_duration = event_maximum_duration

        self.n_samples = len(y_true)
        self.total_duration = ((self.n_samples - 1) * self.sample_offset +
                               self.sample_duration)

        # ROC curve
        roc_fpr, roc_tpr, _ = sk_metrics.roc_curve(y_true, y_pred)
        self.sample_roc_auc = sk_metrics.auc(roc_fpr, roc_tpr)

        # Precision-recall curve
        (prc_precision,
         prc_recall,
         prc_thresholds) = sk_metrics.precision_recall_curve(y_true, y_pred)
        self.sample_prc_auc = sk_metrics.auc(prc_recall, prc_precision)

        f_scores = 2 * (prc_precision * prc_recall) / (
                prc_precision + prc_recall)
        nan_idx = np.argwhere(np.isnan(f_scores))
        f_scores = np.delete(f_scores, nan_idx)
        prc_thresholds = np.delete(prc_thresholds, nan_idx)

        if threshold == "max_f_score":
            self.threshold = float(prc_thresholds[np.argmax(f_scores)])
            self.threshold_method = "max_f_score"
        elif 0 <= float(threshold) <= 1:
            self.threshold = threshold
            self.threshold_method = "fixed"
        else:
            raise ValueError("threshold should be either a number between 0 "
                             "and 1 or 'max_f_score'.")

        y_pred = np.where(y_pred >= self.threshold, 1, 0)
        self.y_pred_post_threshold = y_pred

        # Sample-based metrics
        (self.sample_tn,
         self.sample_fp,
         self.sample_fn,
         self.sample_tp) = sk_metrics.confusion_matrix(y_true, y_pred,
                                                       labels=[0, 1]).ravel()
        
        self.sample_sensitivity = (self.sample_tp /
                                   (self.sample_tp + self.sample_fn))
        self.sample_specificity = (self.sample_tn /
                                   (self.sample_tn + self.sample_fp))
        self.sample_precision = (self.sample_tp /
                                 (self.sample_tp + self.sample_fp))
        self.sample_accuracy = ((self.sample_tp + self.sample_tn) /
                                self.n_samples)
        self.sample_balanced_accuracy = ((self.sample_sensitivity + 
                                          self.sample_specificity) / 2)
        self.sample_recall = self.sample_sensitivity

        self.sample_f_score = sk_metrics.f1_score(y_true, y_pred)

        # Event-based metrics

        # We use the timescoring package, which expects a binary mask of true
        # and predicted events.

        # We first convert y_true and y_pred to masks, where each element is
        # the label for a time duration

        label_duration = sample_offset
        kernel = np.ones(int(sample_duration/label_duration))
        mask_true = np.convolve(y_true, kernel).astype(np.bool_)
        mask_pred = np.convolve(y_pred, kernel).astype(np.bool_)

        annotations_true = Annotation(mask_true, 1/label_duration)
        annotations_pred = Annotation(mask_pred, 1/label_duration)
        self.events_true = annotations_true.events
        self.events_pred = annotations_pred.events

        events_parameters = EventScoring.Parameters(self.event_preictal_tolerance,
                                                    self.event_postictal_tolerance,
                                                    self.event_minimum_rel_overlap,
                                                    self.event_maximum_duration,
                                                    self.event_minimum_separation)
        es = EventScoring(annotations_true, annotations_pred, events_parameters)
        self.events_true_merged_split = es.ref.events
        self.events_pred_merged_split = es.hyp.events

        self.n_true_seizures = es.refTrue

        self.event_tp = es.tp

        # True positive if a predicted event partially overlaps with a true
        # event.
        #detection_delays = list()
        #last_event_extended_end = 0
        #total_ictal_duration = 0
        #for true_event, true_event_extended in zip(self.events_true,
        #                                           self.events_true_extended):
        #    for pred_event in self.events_pred:
        #        overlap = (min(true_event_extended[1], pred_event[1]) -
        #                   max(true_event_extended[0], pred_event[0]))
        #        if overlap >= event_minimum_overlap:
        #            delay = pred_event[0] - true_event[0]
        #            detection_delays.append(delay)
        #
        #    # Sum of ictal duration
        #    # Note: We take into account potential overlap of extended events.
        #    if true_event_extended[0] < last_event_extended_end:
        #        total_ictal_duration += true_event_extended[1] - last_event_extended_end
        #    else:
        #        total_ictal_duration += true_event_extended[1] - true_event_extended[0]
        #    last_event_extended_end = true_event_extended[1]
        
        #if len(detection_delays) > 0:
        #    self.event_average_detection_delay = np.average(detection_delays)
        #else:
        #    self.event_average_detection_delay = np.nan

        self.event_fp = es.fp

        # False positive if a predicted event does not overlap with a true
        # event.
        #self.event_fp = 0
        #for pred_event in self.events_pred:
        #    overlaps = list()
        #    for true_event, true_event_extended in zip(self.events_true,
        #                                               self.events_true_extended):
        #        overlap = (min(true_event_extended[1], pred_event[1]) -
        #                   max(true_event_extended[0], pred_event[0]))
        #        overlaps.append(overlap)
        #
        #    # No positive overlaps means no true event associated to this
        #    # predicted event.
        #    if len(overlaps) > 0:
        #        if max(overlaps) < event_minimum_overlap:
        #            self.event_fp += 1

        self.event_sensitivity = es.sensitivity
        self.event_recall = self.event_sensitivity

        if self.event_tp == 0:                   
            self.event_precision = np.nan                                            
            self.event_f_score = np.nan
        else:
            self.event_precision = (self.event_tp /
                                    (self.event_tp + self.event_fp))
            self.event_f_score = (2 *
                                  (self.event_precision * self.event_recall) /
                                  (self.event_precision + self.event_recall))
        total_duration = (len(y_true) - 1) * sample_offset + sample_duration
        self.event_false_detections_per_hour = ((self.event_fp / total_duration)                
                                                * 3600)

        total_ictal_duration = np.sum(mask_true) * label_duration
        self.event_false_detections_per_interictal_hour = ((self.event_fp / (total_duration -
                                                                             total_ictal_duration) * 3600))

    def as_dict(self) -> dict:
        """
        Returns all non-iterable attributes in a dictionary with attribute
        names as keys. The dict can be used to save attributes to csv, for
        example.
        """
        d = {}

        for field in dataclasses.fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, (str, bool, numbers.Number)):
                d[field.name] = field_value

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

        Parameters
        ----------
        y : np.ndarray
            Array of seizure labels. Expected values are either 0 (no seizure)
            or 1 (seizure).
        sample_duration : float
            Duration of a sample (label) in seconds.
        sample_offset : float 
            Duration between the start of two consecutive samples in seconds.
        event_minimum_separation : float
            Events that are separated by less than event_minimum_separation
            seconds are merged.
        event_maximum_duration : float 
            Events that are longer than event_maximum_duration seconds 
            are split in events with the maximum duration. This is done 
            after the merging of close events (see event_minimum_separation).

        Returns
        -------
        events : List[Tuple[float, float]]
            List of events, where each event is represented by a tuple
            (start_time, end_time), in seconds from the beginning of y.
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
    def _extend_events(events: List[Tuple[float, float]],
                       preictal: float,
                       postictal: float) -> List[Tuple[float, float]]:
        """
        Extend events in the pre- and post-ictal directions.

        The code is adapted from https://github.com/esl-epfl/sz-validation-framework.

        Parameters
        ----------
        events : List[Tuple[float, float]]
            List of events returned by _get_events().
        preictal : float
            Time to extend before each event, in seconds.
        postictal : float
            Time to extend after each event, in seconds.

        Returns
        -------
        extended_events : List[Tuple[float, float]]
            List of extended events, where each event is represented by a tuple
            (start_time, end_time), in seconds from the beginning of y.
        """
        extended_events = events.copy()

        for i, event in enumerate(extended_events):
            extended_events[i] = (event[0] - preictal, event[1] + postictal)

        return extended_events
