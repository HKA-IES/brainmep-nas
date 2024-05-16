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

    Attributes
    ----------
    energy: float
        Energy for a single inference, in joules (J).
    time: float
        Time for a single inference, in seconds (s).
    ram_memory: int
        RAM memory required for the model, in bytes (B).
    flash_memory: int
        Flash memory required for the model, in bytes (B).
    """

    # TODO: Add more details: frequency, voltage, current, hardware, etc.

    energy: float       # in joules (J)
    time: float         # in seconds (s)
    ram_memory: int     # in bytes (B)
    flash_memory: int   # in bytes (B)

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
