import numpy as np
from .fragment import Fragment
from .planet import Planet
import pandas as pd
import logging
import json


logger = logging.getLogger(__name__)


def sanitize_dict(dict: dict, name: str, preserve_time: bool = False) -> dict:
    '''
    sanitize the state dictionary by removing time if needed and adding fragment IDs

    :param name: the fragment name
    :param preserve_time: boolean flag to preserve or discard the time data

    :returns: the new dictionary with sanitized keys
    '''
    new_dict = {}
    for key, value in dict.items():
        if key == 'time' and not preserve_time:
            continue
        new_dict[f'{name}.{key}'] = value

    return new_dict


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
        self.main_body.set_release_properties(0, initial_height, initial_velocity, initial_angle)

        self.fragments: list[Fragment] = []

        logger.info(f"Initializing with diameter: {self.main_body.state.radius * 2}")
        logger.info(f"planet: {planet.name} g: {planet.gravity} m/s^2; Rp: {planet.planet_radius / 1000} km")

    def add_fragment(self, fragment_mass: float, release_pressure: float, C_fr: float = 1.5,
                     alpha: float = 0.0, initial_strength: float = -1) -> None:
        """
        Add a fragment to the model

        :param fragment_mass: mass of the fragment in kg
        :param Prelease: the ram pressure at which the fragment is released from the main body
        :param Cfr: the fragmentation efficiency of the child fragment
        :param alpha: the Weibull strength power of the child fragment
        :param fragment_strength: the optional initial strength of the fragment. set to -1 to borrow the parent strength at time of release
        """
        self.fragments.append(Fragment(len(self.fragments) + 1, fragment_mass,
                                       initial_strength, self.main_body.bulk_density,
                                       self.main_body.ablation_coefficient, C_fr, alpha,
                                       self.planet, release_pressure=release_pressure))

    def save_config(self, filename: str) -> None:
        """
        Save the model configuration to a JSON file

        :param filename: the name of the file to save to
        """
        config = {
            'main_body': self.main_body.get_config(),
            'fragments': [fragment.get_config() for fragment in self.fragments],
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load_config(cls, filename: str, planet: Planet) -> 'FragmentationModel':
        """
        Load the model configuration from a JSON file

        :param filename: the name of the file to load from
        :param planet: the planet object to use for the model
        :return: a FragmentationModel instance
        """
        with open(filename, 'r') as f:
            config = json.load(f)

        return cls.load_from_dict(config, planet)

    @classmethod
    def load_from_dict(cls, config: dict, planet: Planet) -> 'FragmentationModel':
        """
        Load the model configuration from a dictionary

        :param config: the configuration dictionary
        :param planet: the planet object to use for the model
        :return: a FragmentationModel instance
        """

        model = cls(
            **config['main_body'],
            planet=planet
        )

        for i, fragment_config in enumerate(config['fragments']):
            model.add_fragment(
                fragment_mass=fragment_config.pop('initial_mass_fraction') * config['main_body']['initial_mass'],
                **fragment_config
            )

        return model

    def integrate(self, dt: float = 1e-2, max_time: float = 20, min_velocity: float = 100, min_height: float = 100) -> pd.DataFrame:
        """
        Integrate the model forward in time until one of the limits are reached

        :param dt: the timestep in seconds
        :param max_time: the max time for the model in seconds
        :param min_velocity: the minimum velocity to stop computing fragment updates in m/s
        :param min_height: the minimum height at which to stop computation in m

        :returns: pandas DataFrame object with the state variables corresponding to each fragment for each timestep
        """

        t = 0
        times = [0]
        states: list[dict] = []

        # set the initial values for the main body
        self.main_body.release()

        while t < max_time:
            t = times[-1]
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
                    if self.main_body.state.dynamic_pressure > fragment.release_pressure:
                        # if so, release it
                        if fragment.initial_strength == -1:
                            # if the fragment has the same strength as the main body
                            # set it here dynamically
                            fragment.initial_strength = self.main_body.state.strength
                        fragment.set_release_properties(t, self.main_body.state.height, self.main_body.state.velocity,
                                                        np.degrees(self.main_body.state.angle))

                        Mmain = self.main_body.state.mass
                        Mfrag = fragment.initial_mass
                        # recalculate the new radius for the main body by assuming
                        # r(M) = r0(M/M0)^(1/3) (see S(M) from Av 2014)
                        Smain = self.main_body.state.surface_area
                        self.main_body.state.surface_area = Smain * ((Mmain - Mfrag) / Mmain)**(2. / 3.)

                        # update the strength of the main  post fragmentation
                        sigmamain = self.main_body.state.strength
                        alphamain = self.main_body.alpha
                        self.main_body.state.sigma = sigmamain * (Mmain / (Mmain - Mfrag))**(alphamain)

                        # update the mass of the main body
                        self.main_body.state.mass -= fragment.initial_mass
                        fragment.release()
            self.main_body.check_limits(min_velocity, min_height)

            # Update the time
            t += dt
            times.append(t)

            # add the fragment data to a running list
            # first add the main body. we want to use the time from the main body
            state_dict = sanitize_dict(self.main_body.asdict(), 'main', True)

            # then each fragment
            for fragment in self.fragments:
                state_dict = {**state_dict, **sanitize_dict(fragment.asdict(), f'f{fragment.number}')}
            states.append(state_dict)

            # end the sim if all fragments are done
            if self.main_body.done and np.all([fragment.done for fragment in self.fragments]):
                logger.info(f"All fragments reached computation limits at {t:.2f}s")
                break

        # convert to a dataframe and return it
        states_df = pd.DataFrame.from_records(states)
        return states_df
