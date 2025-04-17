from dataclasses import dataclass, field
from .energy import Energy
from .state import State
from .planet import Planet
import numpy as np
import logging

logger = logging.getLogger(__name__)


def surface_area(r):
    return 2 * np.pi * (r**2.)


@dataclass
class Fragment:
    # these are parameters that we define when defining the fragment
    number: int
    initial_mass: float
    initial_strength: float
    bulk_density: float
    ablation_coefficient: float
    C_fr: float
    alpha: float
    planet: Planet
    release_pressure: float = 0

    # these are parameters that will mostly be set only after the fragment is released
    # so these will change with the time/location of release from the main body
    released: bool = field(init=False, default=False)
    done: bool = field(init=False, default=False)
    energy: Energy = field(init=False)
    state: State = field(init=False)

    def __post_init__(self):
        self.state = State()
        self.energy = Energy()

    def get_config(self) -> dict:
        '''
        Return the configuration of the fragment as a dictionary
        '''
        return {
            'number': self.number,
            'initial_mass': self.initial_mass,
            'initial_strength': self.initial_strength,
            'bulk_density': self.bulk_density,
            'ablation_coefficient': self.ablation_coefficient,
            'C_fr': self.C_fr,
            'alpha': self.alpha,
            'release_pressure': self.release_pressure
        }

    def set_release_properties(self, release_time: float, release_altitude: float, release_velocity: float, release_angle: float) -> None:
        """
        Set the release properties of the fragment
        :param release_time: the time at which the fragment was released [s]
        :param release_altitude: the height at which the fragment was released [m]
        :param release_velocity: the velocity at which the fragment was released [m/s]
        :param release_angle: the angle with respect to the vertical [degrees]
        """
        self.release_time = release_time
        self.release_velocity = release_velocity
        self.release_altitude = release_altitude
        self.release_angle = release_angle
        radius = ((3 * self.initial_mass) / (4 * np.pi * self.bulk_density)) ** (1 / 3)
        self.state.radius = radius

    def release(self) -> None:
        """
        Initialize the time-dependent state variables and "release" the fragment
        """
        self.state.mass = self.initial_mass
        self.state.velocity = self.release_velocity
        self.state.angle = np.radians(self.release_angle)
        self.state.time = self.release_time
        self.state.height = self.release_altitude
        self.state.strength = self.initial_strength
        self.state.surface_area = surface_area(self.state.radius)
        self.state.mass_loss_rate = 0
        self.state.acceleration = 0
        self.state.dynamic_pressure = self.planet.rhoz(self.release_altitude) * self.release_velocity ** 2.
        self.state.fragment_count = 1
        self.state.fragment_mass = self.initial_mass
        self.released = True
        self.done = False

        logger.info(f"Releasing fragment {self.number} at time {self.release_time:.2f} s and height {self.release_altitude / 1e3:.2f} km with mass {self.initial_mass / 1e3:.2f} tonnes and velocity {self.release_velocity:.2f} m/s")

    def update(self, dt: float) -> None:
        '''
        Update the fragment state forward in time dt

        :param dt: Timestep in seconds
        '''
        v = self.state.velocity
        M = self.state.mass
        S = self.state.surface_area
        h = self.state.height
        theta = self.state.angle
        Nfr = self.state.fragment_count
        Mfr = self.state.fragment_mass
        sigma = self.state.strength

        logger.debug(f"time: {self.state.time:0.2f} height: {self.state.height:0.2f} mass: {self.state.mass:0.2f} velocity: {self.state.velocity:0.2f}")

        if h > 2501000:
            pass
        elif h < -2498000:
            pass
        else:
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
                Nfr = self.state.fragment_count

            M += dMdt * dt
            Mfr = M / Nfr
            S += dSdt * dt
            theta += dthetadt * dt
            h += -v * np.sin(theta) * dt
            v += dvdt * dt

            # fill in the stuff for the fragment
            self.state.time = self.state.time + dt
            self.state.mass = M
            self.state.velocity = v
            self.state.angle = theta
            self.state.radius = np.sqrt(S / (np.pi))
            self.state.surface_area = S
            self.state.height = h
            self.state.dynamic_pressure = Pram
            self.state.strength = sigma

            self.state.fragment_mass = Mfr
            self.state.fragment_count = Nfr
            self.energy.update(dErdt, dEddt, v, theta)

    def check_limits(self, min_velocity: float, min_height: float) -> None:
        '''
        check the limits of the simulation and set the done flag if the fragment has reached the limits

        :param min_velocity: the minimum velocity to stop computing fragment updates in m/s
        :param min_height: the minimum height at which to stop computation in m
        '''
        if self.state.velocity < min_velocity or self.state.height < min_height:
            logger.info(f"Fragment {self.number} finished at {self.state.time:.2f} s")
            self.done = True

    def asdict(self) -> dict[float]:
        '''
        convert the dataclass to a dictionary
        '''

        return {**self.state.asdict(), **self.energy.asdict()}
