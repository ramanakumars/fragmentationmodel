from dataclasses import dataclass, field
import numpy as np


@dataclass()
class Energy:
    total: list[float] = field(init=False)  # total power in W
    radiated: list[float] = field(init=False)  # total radiated power in W
    deposited: list[float] = field(init=False)  # total deposited power in J/km

    def __post_init__(self):
        self.total = [0]
        self.radiated = [0]
        self.deposited = [0]

    def append(self, dErdt, dEddt, v, theta):
        """
        Add the energy data to the list

        :param dErdt: radiated power in W
        :param dEddt: deposited power in W
        :param v: velocity in m/s
        :param theta: angle in radians
        """
        self.radiated.append(dErdt)
        self.deposited.append(dEddt / (v * np.sin(theta)))
        self.total.append(dErdt + dEddt)

    def convert_to_arrays(self) -> None:
        """
        Helper function for converting everything to arrays
        """
        for key in ['total', 'radiated', 'deposited']:
            self.__setattr__(key, np.asarray(self.__getattribute__(key)))
