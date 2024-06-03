# -*- coding: utf-8 -*-

# import built-in module
import dataclasses
import numbers

# import third-party modules

# import your own module


@dataclasses.dataclass
class HardwareMetrics:
    """
    Store hardware metrics.

    Note: The attributes defined here are only the bare minimum required to
    characterize the hardware performance of a model on a specific hardware.

    Attributes
    ----------
    inference_time: float
        Time for a single inference, in seconds (s).
    inference_energy: float
        Energy for a single inference, in joules (J).
    """
    inference_time: float
    inference_energy: float

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
