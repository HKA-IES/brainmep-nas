# -*- coding: utf-8 -*-

# import built-in module
import dataclasses
import numbers
from typing import Optional

# import third-party modules

# import your own module
from .accuracymetrics import AccuracyMetrics
from .hardwaremetrics import HardwareMetrics


@dataclasses.dataclass
class CombinedMetrics:
    """
    Store metrics which are obtained from combination of accuracy and hardware
    metrics.

    Attributes
    ----------
    accuracy_metrics: AccuracyMetrics
        AccuracyMetrics object used as input.
    hardware_metrics: HardwareMetrics
        HardwareMetrics object used as input.
    stimulation_energy: float
        Energy required for a single stimulation, in joules (J).
    inference_duty_cycle: float
        Proportion of the time spent in inference with respect to the cycle
        time (time between the beginning of two consecutive inferences).
            inference_duty_cycle = inference_time / sample_offset
    total_inference_energy: float
        Total inference energy, in joules (J).
    inference_energy_per_hour: float
        Inference energy per hour, in joules per hour (J/h).
    event_tp_total_stimulation_energy: float, optional
        Total stimulation energy if a stimulation is done for each true
        positive event, in joules (J).
        Note: Calculated only if stimulation_energy != None.
    event_tp_stimulation_energy_per_hour: float, optional
        Stimulation energy per hour if a stimulation is done for each true
        positive event, in joules per hour (J/h).
        Note: Calculated only if stimulation_energy != None.
    event_fp_total_stimulation_energy: float, optional
        Total stimulation energy if a stimulation is done for each false
        positive event, in joules (J).
        Note: Calculated only if stimulation_energy != None.
    event_fp_stimulation_energy_per_hour: float, optional
        Stimulation energy per hour if a stimulation is done for each false
        positive event, in joules per hour (J/h).
        Note: Calculated only if stimulation_energy != None.
    event_total_stimulation_energy: float, optional
        Total stimulation energy if a stimulation is done for each predicted
        event (TP + FP), in joules (J).
        Note: Calculated only if stimulation_energy != None.
    event_stimulation_energy_per_hour: float, optional
        Stimulation energy per hour if a stimulation is done for each predicted
        event (TP + FP), in joules per hour (J/h).
        Note: Calculated only if stimulation_energy != None.
    total_undesired_energy: float, optional
        Sum of the total inference energy and the total stimulation energy due
        to false positives. We exclude the stimulation energy due to true
        positives because this stimulation is a desired effect of the system.
            total_undesired_energy = event_fp_total_stimulation_energy +
                                     total_inference_energy
        Note: Calculated only if stimulation_energy != None.
    undesired_energy_per_hour: float, optional
        Undesired energy per hour, in joules per hour (J/h).
        Note: Calculated only if stimulation_energy != None.
    """

    # Input data
    accuracy_metrics: AccuracyMetrics
    hardware_metrics: HardwareMetrics
    stimulation_energy: Optional[float]

    # Combined metrics
    inference_duty_cycle: float
    total_inference_energy: float
    inference_energy_per_hour: float

    # Combined metrics depending on stimulation_energy
    event_tp_total_stimulation_energy: Optional[float]
    event_tp_stimulation_energy_per_hour: Optional[float]
    event_fp_total_stimulation_energy: Optional[float]
    event_fp_stimulation_energy_per_hour: Optional[float]
    event_total_stimulation_energy: Optional[float]
    event_stimulation_energy_per_hour: Optional[float]
    total_undesired_energy: Optional[float]
    undesired_energy_per_hour: Optional[float]

    def __init__(self, accuracy_metrics: AccuracyMetrics,
                 hardware_metrics: HardwareMetrics,
                 stimulation_energy: Optional[float] = None):
        """
        Calculate metrics which are a combination of accuracy and hardware
        metrics.

        Parameters
        ----------
        accuracy_metrics: AccuracyMetrics
            AccuracyMetrics object.
        hardware_metrics: HardwareMetrics
            HardwareMetrics object.
        stimulation_energy: float, optional
            Energy for a single brain stimulation event, in joules (J).
        """
        am = accuracy_metrics
        hm = hardware_metrics
        self.accuracy_metrics = accuracy_metrics
        self.hardware_metrics = hardware_metrics
        self.stimulation_energy = stimulation_energy

        self.inference_duty_cycle = hm.inference_time / am.sample_offset
        self.total_inference_energy = hm.inference_energy * am.n_samples
        self.inference_energy_per_hour = (self.total_inference_energy *
                                          (3600 / am.total_duration))

        # Metrics depending on stimulation_energy
        if self.stimulation_energy is not None:
            self.event_tp_total_stimulation_energy = (am.event_tp *
                                                      stimulation_energy)
            self.event_tp_stimulation_energy_per_hour = (
                    self.event_tp_total_stimulation_energy *
                    (3600 / am.total_duration))
            self.event_fp_total_stimulation_energy = (am.event_fp *
                                                      stimulation_energy)
            self.event_fp_stimulation_energy_per_hour = (
                    self.event_fp_total_stimulation_energy *
                    (3600 / am.total_duration))
            self.event_total_stimulation_energy = (
                    self.event_tp_total_stimulation_energy +
                    self.event_fp_total_stimulation_energy)
            self.event_stimulation_energy_per_hour = (
                    self.event_tp_stimulation_energy_per_hour +
                    self.event_fp_stimulation_energy_per_hour)
            self.total_undesired_energy = (
                    self.event_fp_total_stimulation_energy +
                    self.total_inference_energy)
            self.undesired_energy_per_hour = (
                    self.event_fp_stimulation_energy_per_hour +
                    self.inference_energy_per_hour)
        else:
            self.event_tp_total_stimulation_energy = None
            self.event_tp_stimulation_energy_per_hour = None
            self.event_fp_total_stimulation_energy = None
            self.event_fp_stimulation_energy_per_hour = None
            self.event_total_stimulation_energy = None
            self.event_stimulation_energy_per_hour = None
            self.total_undesired_energy = None
            self.undesired_energy_per_hour = None

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
