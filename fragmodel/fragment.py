from dataclasses import dataclass, field
from .energy import Energy
from .planet import Planet
import numpy as np
import copy
import logging

logger = logging.getLogger(__name__)


def surface_area(r):
    return np.pi * (r**2.)


@dataclass
class Fragment:
    number: int
    initial_mass: float
    initial_strength: float
    bulk_density: float
    ablation_coefficient: float
    C_fr: float
    alpha: float
    planet: Planet
    release_pressure: float = 0

    released: bool = field(init=False, default=False)
    done: bool = field(init=False, default=False)
    energy: Energy = field(init=False)
    time: list[float] = field(init=False)
    angle: list[float] = field(init=False)
    mass: list[float] = field(init=False)
    sigma: list[float] = field(init=False)
    radius: list[float] = field(init=False)
    surface_area: list[float] = field(init=False)
    velocity: list[float] = field(init=False)
    mass_loss_rate: list[float] = field(init=False)
    acceleration: list[float] = field(init=False)
    dynamic_pressure: list[float] = field(init=False)
    fragment_count: list[float] = field(init=False)
    fragment_mass: list[float] = field(init=False)

    def release(self, release_time: float, release_altitude: float, release_velocity: float, release_angle: float) -> None:
        """
        Initialize the time-dependent state variables and "release" the fragment

        :param release_time: the time at which the fragment was released [s]
        :param release_altitude: the height at which the fragment was released [m]
        :param release_velocity: the velocity at which the fragment was released [m/s]
        :param release_angle: the angle with respect to the vertical [degrees]
        """
        self.time = [release_time]
        self.mass = [self.initial_mass]
        self.velocity = [release_velocity]
        self.angle = [np.radians(release_angle)]
        self.sigma = [self.initial_strength]
        self.height = [release_altitude]
        radius = ((3 * self.initial_mass) / (4 * np.pi * self.bulk_density)) ** (1 / 3)
        self.radius = [radius]
        self.surface_area = [surface_area(radius)]
        self.mass_loss_rate = [0]
        self.acceleration = [0]
        self.dynamic_pressure = [self.planet.rhoz(release_altitude) * release_velocity ** 2.]
        self.fragment_count = [1]
        self.fragment_mass = [self.initial_mass]
        self.released = True
        self.done = False
        self.energy = Energy()

        logger.info(f"Releasing fragment {self.number} at time {release_time} s and height {release_altitude} m with mass {self.initial_mass} kg and velocity {release_velocity} m/s")

    def update(self, dt: float) -> None:
        '''
        Update the fragment state forward in time dt

        :param dt: Timestep in seconds
        '''
        v = copy.deepcopy(self.velocity[-1])
        M = copy.deepcopy(self.mass[-1])
        S = copy.deepcopy(self.surface_area[-1])
        h = copy.deepcopy(self.height[-1])
        theta = copy.deepcopy(self.angle[-1])
        Nfr = copy.deepcopy(self.fragment_count[-1])
        Mfr = copy.deepcopy(self.fragment_mass[-1])
        sigma = copy.deepcopy(self.sigma[-1])

        logger.debug(f"time: {self.time[-1]:0.2f} height: {self.height[-1]:0.2f} mass: {self.mass[-1]:0.2f} velocity: {self.velocity[-1]:0.2f}")

        rho_a = self.planet.rhoz(h)
        Pram = rho_a * (v**2.)

        dvdt = -self.planet.Cd * S * rho_a * (v**2.) / (2. * M) + self.planet.gravity * np.sin(theta)
        dMdt = -self.ablation_coefficient * S * rho_a * (np.abs(v)**3.)

        dthetadt = self.planet.gravity * np.cos(theta) / v - v * np.sin(theta) / (self.planet.planet_radius + h)

        dEdtd = self.planet.Cd * S * rho_a * (np.abs(v)**3.) / 2.
        dEdta = self.ablation_coefficient * S * rho_a * (np.abs(v)**5.) / (2.)

        dEtdt = dEdtd + dEdta  # released energy
        dErdt = self.planet.Cr * dEdta  # radiated energy
        dEddt = dEtdt - dErdt  # deposited energy
        dSdt = (2. / 3.) * (S / M) * dMdt

        if (Pram > sigma):
            Cfr = self.C_fr
            s0 = self.initial_strength
            M0 = self.initial_mass
            alpha = self.alpha

            dSdtfr = Cfr * np.sqrt(Pram - sigma) / (M**(1. / 3.) * self.bulk_density**(1. / 6.)) * S
            dSdt += dSdtfr

            Nfr = 16. * (S**3.) * self.bulk_density**2. / (9. * np.pi * M**2.)
            sigma = s0 * (M0 / Mfr)**(alpha)
        else:
            Nfr = 1

        M += dMdt * dt
        Mfr = M / Nfr
        S += dSdt * dt
        theta += dthetadt * dt
        h += -v * np.sin(theta) * dt
        v += dvdt * dt

        ''' fill in the stuff for the fragment '''
        self.time.append(self.time[-1] + dt)
        self.mass.append(M)
        self.velocity.append(v)
        self.angle.append(theta)
        self.radius.append(np.sqrt(S / (np.pi)))
        self.surface_area.append(S)
        self.height.append(h)
        self.dynamic_pressure.append(Pram)
        self.sigma.append(sigma)

        self.fragment_mass.append(Mfr)
        self.fragment_count.append(Nfr)
        self.energy.append(dErdt, dEddt, v, theta)

    def check_limits(self, min_velocity: float, min_height: float):
        if self.velocity[-1] < min_velocity or self.height[-1] < min_height:
            logger.info(f"Fragment {self.number} finished at {self.time[-1]} s")
            self.done = True

    def convert_to_arrays(self) -> None:
        """
        Helper function for converting everything to arrays
        """
        for key in ['time', 'mass', 'velocity', 'angle', 'radius', 'surface_area',
                    'height', 'dynamic_pressure', 'sigma', 'fragment_mass', 'fragment_count']:
            self.__setattr__(key, np.asarray(self.__getattribute__(key)))

        self.energy.convert_to_arrays()
