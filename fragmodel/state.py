from dataclasses import dataclass, field, fields, asdict
import numpy as np


@dataclass
class State:
    """
    Holds the state of the fragment at a given time
    """

    time: float = field(init=False)
    """ the time corresponding to this state [s]"""
    angle: float = field(init=False)
    """ the angle of the fragment with respect to the horizontal plane [degrees]"""
    mass: float = field(init=False)
    """ the current mass of the fragment [kg]"""
    strength: float = field(init=False)
    """ the current strength of the fragment [Pa]"""
    height: float = field(init=False)
    """ the current height of the fragment [m]"""
    radius: float = field(init=False)
    """ the current radius of the fragment [m]"""
    surface_area: float = field(init=False)
    """ the current surface area of the fragment [m^2]"""
    velocity: float = field(init=False)
    """ the current velocity of the fragment [m/s]"""
    mass_loss_rate: float = field(init=False)
    """ the current mass loss rate of the fragment [kg/s]"""
    acceleration: float = field(init=False)
    """ the current deceleration of the fragment [m/s^2]"""
    dynamic_pressure: float = field(init=False)
    """ the current dynamic pressure of the fragment [Pa]"""
    fragment_count: float = field(init=False)
    """ the number of sub-fragments [unitless]"""
    fragment_mass: float = field(init=False)
    """ the mass of individual sub-fragments [kg] (count * fragment_mass = mass)"""

    def __post_init__(self):
        '''
        initialize all values to nan initially
        '''
        fields_list = fields(self)
        for field_i in fields_list:
            setattr(self, field_i.name, np.nan)

    def asdict(self) -> dict[float]:
        '''
        convert the dataclass to a dictionary

        :returns: the dictionary containing the dataclass fields
        '''
        return {
            key: value for key, value in asdict(self).items()
        }

    @staticmethod
    def colnames() -> list[str]:
        '''
        get the field names as a list
        '''
        return list(field.name for field in fields(State))
