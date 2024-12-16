import numpy as np
from .fragment import Fragment
from .planet import Planet
import logging


logger = logging.getLogger(__name__)


class FragmentationModel:
    def __init__(self, initial_mass: float, initial_velocity: float, initial_angle: float, initial_height: float,
                 strength: float, ablation_coefficient: float, bulk_density: float,
                 C_fr: float, alpha: float, planet: Planet):
        '''
        initialize values for the main body

        Parameters
        ----------
        :param initial_mass: mass of the main body (kg)
        :param initial_velocity: impacting velocity (m/s)
        :param initial_angle: angle with respect to the horizontal plane (degrees)
        :param iniital_height: initial height (m)
        :param strength: initial bulk strength of the main body (Pa)
        :param ablation_coefficient: ablation coefficient (kg/J)
        :param bulk_density: bulk density of the meteor (kg/m^3)
        :param Cfr: fragment parameter (see Avramenko et al., 2017, unitless)
        :param alpha: Weibull strength scaling parameter (unitless)
        '''
        self.planet = planet

        self.main_body = Fragment(0, initial_mass, strength, bulk_density, ablation_coefficient, C_fr,
                                  alpha, self.planet)

        self.main_body.release(0, initial_height, initial_velocity, initial_angle)

        self.fragments: list[Fragment] = []

        logger.info(f"Initializing with diameter: {self.main_body.radius[0] * 2}")
        logger.info(f"planet: {planet.name} g: {planet.gravity} m/s^2; Rp: {planet.planet_radius / 1000} km")

    def add_fragment(self, fragment_mass: float, Prelease: float, Cfr: float = 1.5,
                     alpha: float = 0.0, fragment_strength: float = -1) -> None:
        """
        Add a fragment to the model

        :param fragment_mass: mass of the fragment in kg
        :param Prelease: the ram pressure at which the fragment is released from the main body
        :param Cfr: the fragmentation efficiency of the child fragment
        :param alpha: the Weibull strength power of the child fragment
        :param fragment_strength: the optional initial strength of the fragment. set to -1 to borrow the parent strength at time of release
        """
        self.fragments.append(Fragment(len(self.fragments) + 1, fragment_mass,
                                       fragment_strength, self.main_body.bulk_density,
                                       self.main_body.ablation_coefficient, Cfr, alpha,
                                       self.planet, release_pressure=Prelease))

    def integrate(self, dt: float = 1e-2, max_time: float = 20, min_velocity: float = 100,
                  min_height: float = 100):
        """
        Integrate the model forward in time until one of the limits are reached

        :param dt: the timestep in seconds
        :param max_time: the max time for the model in seconds
        :param min_velocity: the minimum velocity to stop computing fragment updates in m/s
        :param min_height: the minimum height at which to stop computation in m
        """

        t = 0

        while t < max_time:
            # update the main body first
            # this involves calculating the next timestep in the position, velocity and mass
            # of the main body
            self.main_body.update(dt)
            for fragment in self.fragments:
                # do the same for the released fragments
                if fragment.released and not fragment.done:
                    fragment.update(dt)
                    fragment.check_limits(min_velocity, min_height)
                elif not fragment.released:
                    # for the fragments that are not released, check if the main body's
                    # ram pressure exceeds the release pressure
                    if self.main_body.dynamic_pressure[-1] > fragment.release_pressure:
                        # if so, release it
                        if fragment.initial_strength == -1:
                            # if the fragment has the same strength as the main body
                            # set it here dynamically
                            fragment.initial_strength = self.main_body.sigma[-1]
                        fragment.release(t, self.main_body.height[-1], self.main_body.velocity[-1],
                                         np.degrees(self.main_body.angle[-1]))

                        Mmain = self.main_body.mass[-1]
                        Mfrag = fragment.initial_mass
                        # recalculate the new radius for the main body by assuming
                        # r(M) = r0(M/M0)^(1/3) (see S(M) from Av 2014)
                        Smain = self.main_body.surface_area[-1]
                        self.main_body.surface_area[-1] = Smain * ((Mmain - Mfrag) / Mmain)**(2. / 3.)

                        # update the strength of the main  post fragmentation
                        sigmamain = self.main_body.sigma[-1]
                        alphamain = self.main_body.alpha
                        self.main_body.sigma[-1] = sigmamain * (Mmain / (Mmain - Mfrag))**(alphamain)

                        # update the mass of the main body
                        self.main_body.mass[-1] -= fragment.initial_mass

            # Update the time
            t += dt
        self.main_body.convert_to_arrays()
        for fragment in self.fragments:
            fragment.convert_to_arrays()
