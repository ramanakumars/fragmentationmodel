from dataclasses import dataclass, field, fields, asdict
import numpy as np


@dataclass
class State:
    time: float = field(init=False)
    angle: float = field(init=False)
    mass: float = field(init=False)
    strength: float = field(init=False)
    height: float = field(init=False)
    radius: float = field(init=False)
    surface_area: float = field(init=False)
    velocity: float = field(init=False)
    mass_loss_rate: float = field(init=False)
    acceleration: float = field(init=False)
    dynamic_pressure: float = field(init=False)
    fragment_count: float = field(init=False)
    fragment_mass: float = field(init=False)

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
