from dataclasses import asdict, dataclass, field, fields

import numpy as np


@dataclass()
class Energy:
    """
    Class to hold the energy release information for the fragment at a given time
    """

    total: float = field(init=False)
    """ total power in W"""
    radiated: float = field(init=False)
    """ total radiated power in W """
    deposited: float = field(init=False)
    """ total deposited power in J/km"""

    def __post_init__(self):
        """
        initialize the arrays
        """
        self.total = 0
        self.radiated = 0
        self.deposited = 0

    def update(self, dErdt: float, dEddt: float, v: float, theta: float) -> None:
        """
        Add the energy data to the list

        :param dErdt: radiated power in W
        :param dEddt: deposited power in W
        :param v: velocity in m/s
        :param theta: angle in radians
        """
        self.radiated = dErdt
        self.deposited = dEddt / (v * np.sin(theta))
        self.total = dErdt + dEddt

    def asdict(self) -> dict[float]:
        '''
        convert the dataclass to a dictionary
        '''
        return {key: value for key, value in asdict(self).items()}

    @staticmethod
    def colnames() -> list[str]:
        return list(field.name for field in fields(Energy))
