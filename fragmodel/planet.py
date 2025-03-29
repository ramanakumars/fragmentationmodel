from dataclasses import dataclass
import numpy as np
import pathlib
from scipy.interpolate import interp1d


@dataclass(init=False)
class Planet:
    """
    Class to define the planetary constants and the temperature/pressure profile
    """

    name: str
    """ name of the planet """
    gravity: float
    """ gravity of the planet [m/s^2] """
    Cd: float
    """ drag coefficient of the planet [unitless] """
    planet_radius: float
    """ planet radius [m] """
    Cr: float
    """ ratio of ablation released as heat [unitless] """
    Ratmo: float
    """ dry specific gas constant [J/(kg*K)] """
    rhoz: callable
    """ function to calculate the density as a function of height [kg/m^3] """
    Pz: callable
    """ function to calculate the pressure as a function of height [Pa] """

    def __init__(self, planet: str):
        """
        define the planetary constants based on the planet selected

        :param planet: the name of the planet
        """
        self.Cr = 0.37  # ratio of ablation released as heat (Av 2014)
        if planet.lower() == "earth":
            self.name = "Earth"
            self.Cd = 0.75
            self.gravity = 9.81  # gravity
            self.planet_radius = 6371000.  # planet radius
            self.Ratmo = 287.  # dry specific gas constant
        elif planet.lower() == "jupiter":
            self.name = "Jupiter"
            self.Cd = 0.92  # from Carter, Jandir & Kress results in 2009 LPSC
            self.gravity = 24.00  # gravity
            self.planet_radius = 70000000  # planet radius
            self.Ratmo = 3637.  # dry specific gas constant
        else:
            raise NotImplementedError("Planet is not implemented")

    def define_temperature_profile(self, tpzfile: pathlib.Path) -> None:
        """
        Loading the temperature/pressure profile as a function of height and set the
        corresponding density/pressure functions

        :param tpzfile: path to the altitude/pressure/temperature file
        """
        # Get TP profile
        tpz = np.genfromtxt(tpzfile, skip_header=1, delimiter=',')

        # Interpolate the tpz profile
        zd = tpz[:, 0]  # in km
        Pd = tpz[:, 1]  # in mbar
        Td = tpz[:, 2]  # in K

        Pz = interp1d(zd * 1000., np.log10(Pd), kind='cubic')
        Tz = interp1d(zd * 1000., Td, kind='cubic')

        self.rhoz = lambda z: 10.**(Pz(z) + 2.) / (self.Ratmo * Tz(z))
        self.Pz = lambda z: 10.**(Pz(z) + 2.)
