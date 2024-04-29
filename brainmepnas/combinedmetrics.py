# -*- coding: utf-8 -*-

# import built-in module
import dataclasses
import numbers

# import third-party modules

# import your own module
from .accuracymetrics import AccuracyMetrics
from .hardwaremetrics import HardwareMetrics


@dataclasses.dataclass
class CombinedMetrics:
    """
    Store metrics which are obtained from combination of accuracy and hardware
    metrics.
    """

    # Input data
    accuracy_metrics: AccuracyMetrics
    hardware_metrics: HardwareMetrics
    stimulation_energy: float       # in joules (J)

    # Combined metrics
    busy_cycle: float       # proportion of the time between cycles which is
                            # taken for an inference. << 100% is desired.
    inference_energy_per_hour: float                # in joules (J)
    event_tp_stimulation_energy_per_hour: float     # in joules (J)
    event_fp_stimulation_energy_per_hour: float     # in joules (J)
    event_stimulation_energy_per_hour: float        # in joules (J)
    undesired_energy_per_hour: float                # energy from interference
                                                    # + false stimulation,
                                                    # in joules (J)

    def __init__(self, accuracy_metrics: AccuracyMetrics,
                 hardware_metrics: HardwareMetrics,
                 stimulation_energy: float):
        """
        :param accuracy_metrics: AccuracyMetrics instance.
        :param hardware_metrics: HardwareMetrics instance.
        :param stimulation_energy: energy per stimulation, in joules.
        """
        am = accuracy_metrics
        hm = hardware_metrics
        self.accuracy_metrics = accuracy_metrics
        self.hardware_metrics = hardware_metrics
        self.stimulation_energy = stimulation_energy

        self.busy_cycle = hm.time / am.sample_offset
        self.inference_energy_per_hour = (3600 / am.sample_offset) * hm.energy
        self.event_tp_stimulation_energy_per_hour = (am.event_tp / am.total_duration) * 3600 * self.stimulation_energy
        self.event_fp_stimulation_energy_per_hour = (am.event_fp / am.total_duration) * 3600 * self.stimulation_energy
        self.event_stimulation_energy_per_hour = (self.event_tp_stimulation_energy_per_hour +
                                                  self.event_fp_stimulation_energy_per_hour)
        self.undesired_energy_per_hour = (self.inference_energy_per_hour +
                                          self.event_fp_stimulation_energy_per_hour)

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
