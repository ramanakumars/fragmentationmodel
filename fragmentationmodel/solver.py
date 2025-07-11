import logging
from collections import OrderedDict
from copy import deepcopy
from multiprocessing import Pool

import emcee
import numpy as np

from .fragmentation_model import FragmentationModel
from .planet import Planet

logger = logging.getLogger(__name__)

# 1 kt in joules
kt = 4.184e12


def normalize(parameter: float, min: float, max: float) -> float:
    '''
    Normalize the parameter to be between 0 and 1

    :param parameter: the parameter to normalize
    :param min: the minimum value of the parameter
    :param max: the maximum value of the parameter

    :returns: the normalized parameter
    '''
    return (parameter - min) / (max - min)


def denormalize(parameter: float, min: float, max: float) -> float:
    '''
    Denormalize the parameter to be between min and max

    :param parameter: the normalized parameter
    :param min: the minimum value of the parameter
    :param max: the maximum value of the parameter

    :returns: the denormalized parameter
    '''
    return (max - min) * parameter + min


def update_dict(orig_dict: dict, new_dict: dict) -> dict:
    """
    Update the original dictionary with the new dictionary recursively
    :param orig_dict: the original dictionary
    :param new_dict: the new dictionary

    :returns: the updated dictionary
    """
    for key, value in new_dict.items():
        if isinstance(value, dict):
            orig_dict[key] = update_dict(orig_dict.get(key, {}), value)
        else:
            orig_dict[key] = value
    return orig_dict


def params_to_dict(p: np.ndarray, parameters: dict) -> dict:
    """
    Convert the parameters from the MCMC sampler to an input config format for the frag model
    :param p: the parameters from the MCMC sampler
    :param parameters: the parameters dictionary from the model (containing the locations key for each parameter)

    :returns: the updated config dictionary for the frag model
    """
    locations: dict[str] = {key: value['location'] for key, value in parameters.items()}

    updated_config = {'main_body': {}, 'fragments': []}

    fragments = {}

    for i, (key, location) in enumerate(locations.items()):
        # the main body is stored as `main_body.<key>` in the config
        if 'main_body' in location:
            dict_key = location.split('.')[1]
            updated_config['main_body'][dict_key] = denormalize(
                p[key], parameters[key]['min'], parameters[key]['max']
            )

        # the fragments are stored as `fragment.<index>.<key>` in the config
        elif 'fragment' in location:
            # parse the fragment index and key from the location
            _, fragment_index, fragment_key = location.split('.')

            # first fragment is index 1 since the main body is technically index 0
            fragment_index = int(fragment_index) - 1
            if fragment_index not in fragments:
                fragments[fragment_index] = {}
            fragments[fragment_index][fragment_key] = denormalize(
                p[key], parameters[key]['min'], parameters[key]['max']
            )

    # sort the fragments by index
    fragments = OrderedDict(sorted(fragments.items()))

    # and then update the config with the fragments
    # in the order of their index
    for fragment in fragments.values():
        updated_config['fragments'].append(fragment)

    return updated_config


class MCMCSolver:
    def __init__(self, base_config: dict, planet: Planet):
        self.base_config = base_config
        self.planet = planet

    def set_parameters(self, parameters: dict):
        self.parameters = parameters
        self.parameter_names = list(parameters.keys())
        self.ndims = len(self.parameter_names)

    def set_lightcurve(
        self,
        dependent_axis: np.ndarray,
        lightcurve: np.ndarray,
        lightcurve_err: np.ndarray | float,
        lightcurve_type: str = 'lightcurve',
    ):
        '''
        Set the observe lightcurve for fitting

        :param dependent_axis: the input x-dimension (either time [s] or height [m], see `lightcurve_type`)
        :param lightcurve: the input lightcurve (either W or kt/km, see `lightcurve_type`)
        :param lightcurve_err: error in lightcurve (constant if equal for all time or array for each measurement, same units as `lightcurve`)
        :param lightcurve_type: the type of lightcurve (either 'lightcurve' or 'energy_deposition'). If passing in a lightcurve, `dependent_axis` should be time (s) and `lightcurve` should be in W. If passing in energy deposition, `dependent_axis` should be height (m) and `lightcurve` should be in W/km.

        :raises AssertionError: if input spectrum wavelength range is outside the model config wavelength range
        '''
        if lightcurve_type.lower() not in ['lightcurve', 'energy_deposition']:
            raise ValueError(
                "lightcurve_type must be either 'lightcurve' or 'energy_deposition'"
            )

        self.ref_lightcurve = lightcurve
        self.ref_dep_axis = dependent_axis
        self.ref_lightcurve_error = lightcurve_err * self.ref_lightcurve
        self.lightcurve_type = lightcurve_type

    def get_initial_guess(self, n_walkers: int) -> np.ndarray:
        '''
        Get the initial guess for the MCMC walkers

        :param n_walkers: the number of walkers to use

        :returns: the initial guess for the MCMC walkers
        '''
        # get a random sample of the parameter ranges
        initial_guess = np.zeros((n_walkers, self.ndims))
        for i, key in enumerate(self.parameter_names):
            initial_guess[:, i] = np.random.uniform(0, 1, n_walkers)

        return initial_guess

    def set_integration_parameters(self, integration_parameters: dict):
        '''
        Set the integration parameters for the model

        :param integration_parameters: the integration parameters to use for the model
        '''
        self.integration_parameters = integration_parameters

    def run_mcmc(
        self,
        num_steps: int = 150,
        n_walkers: int = 10,
        verbose: bool = True,
        threads: int = 1,
    ):
        '''
        Run the MCMC sampler and fit the parameters

        :param num_steps: the number of steps to run the walkers (default=150)
        :param n_walkers: the number of walkers to use (default=10)
        :param threads: the number of multiprocessing threads to use (must be <= n_walkers, default = 6)
        :param verbose: for progress verbosity (default = true)

        :returns: the emcee EnsembleSampler object with the walker history
        '''
        initial_guess = self.get_initial_guess(n_walkers)
        with Pool(processes=threads) as pool:
            self.sampler = emcee.EnsembleSampler(
                n_walkers,
                self.ndims,
                log_likelihood,
                args=(
                    self.parameters,
                    self.base_config,
                    self.planet,
                    self.integration_parameters,
                    self.ref_lightcurve,
                    self.ref_dep_axis,
                    self.ref_lightcurve_error,
                    self.lightcurve_type,
                ),
                parameter_names=self.parameter_names,
                pool=pool,
            )
            return self.sampler.run_mcmc(initial_guess, num_steps, progress=verbose)

    def get_new_config(self, p: np.ndarray | dict) -> dict:
        '''
        Get the new configuration for the model based on the current parameters

        :param p: the current parameters

        :returns: the new configuration for the model
        '''

        if isinstance(p, np.ndarray):
            p = dict(zip(self.parameter_names, p))
        elif not isinstance(p, dict):
            raise ValueError("p must be either a numpy array or a dictionary")

        return FragmentationModel.load_from_dict(
            update_dict(deepcopy(self.base_config), params_to_dict(p, self.parameters)),
            self.planet,
        )


def likelihood_prior(p: dict[float]) -> float:
    '''
    Get the model prior for each parameters.
    This is essentially a wrapper function to set a prior of -infinity when the parameter is outside its defined range

    :param p: list containing current values of the parameters
    :param parameters: dict containing min/max values for each parameter

    :returns: the log-probability for the prior
    '''
    check = [(param > 0) & (param < 1) for key, param in p.items()]
    if np.all(check):
        return 0.0
    else:
        return -np.inf


def log_likelihood(
    p: dict[float],
    parameters: dict,
    base_config: dict,
    planet: Planet,
    integration_parameters: dict,
    ref_lightcurve: np.ndarray,
    ref_dep_axis: np.ndarray,
    ref_lightcurve_error: np.ndarray,
    lightcurve_type: str,
) -> float:
    '''
    Get the log-likelihood for the current set of parameters by running FragmentationModel and comparing the lightcurve.
    This is what emcee will use to find the global minima. Returns -inf if any parameter in `p` is out of range.

    :param p: current test values of the model parameters
    :param parameters: dict containing information about each parameter (min/max values, location in config, etc.)
    :param base_config: the base configuration for the fragmentation model. The parameters in `p` will be used to update this config.
    :param planet: the planet object to use for the model (see `Planet`)
    :param integration_parameters: the integration parameters to use for the model (see `FragmentationModel.integrate`)
    :param ref_lightcurve: the reference lightcurve to compare against
    :param ref_dep_axis: the reference dependent axis (either time [s] or height [m], see `lightcurve_type`)
    :param ref_lightcurve_error: the error in the reference lightcurve (constant if equal for all time or array for each measurement, same units as `ref_lightcurve`)
    :param lightcurve_type: the type of lightcurve (either 'lightcurve' or 'energy_deposition'). If passing in a lightcurve, `ref_dep_axis` should be time (s) and `ref_lightcurve` should be in W. If passing in energy deposition, `ref_dep_axis` should be height (m) and `ref_lightcurve` should be in kt/km.

    :returns: the log-likelihood value for the current set of parameters `p`
    '''
    ln_prior = likelihood_prior(p)
    if not np.isfinite(ln_prior):
        return -np.inf

    updated_config = params_to_dict(p, parameters)

    model = FragmentationModel.load_from_dict(
        update_dict(deepcopy(base_config), updated_config), planet
    )

    # check whether the height goes out of bounds
    try:
        df = model.integrate(**integration_parameters)
    except ValueError as e:
        logger.warning(e)
        return -np.inf

    if lightcurve_type == 'lightcurve':
        lightcurve = df['main.total']
        for i in range(len(model.fragments)):
            lightcurve += df[f'f{i + 1}.total']

        # shift the model lightcurve to match the reference lightcurve by aligning the lightcurve peak
        t_peak_model = df['main.time'][np.argmax(lightcurve)]
        t_peak_ref = ref_dep_axis[np.argmax(ref_lightcurve)]

        model_lightcurve_interped = np.interp(
            ref_dep_axis - t_peak_ref,
            df['main.time'] - t_peak_model,
            lightcurve,
            left=0,
            right=0,
        )

    elif lightcurve_type == 'energy_deposition':
        model_lightcurve_interped = np.interp(
            ref_dep_axis,
            df['main.height'][::-1],
            df['main.deposited'][::-1] * (1000 / kt),
            left=0,
            right=0,
        )
        for i in range(len(model.fragments)):
            model_lightcurve_interped += np.interp(
                ref_dep_axis,
                df[f'f{i + 1}.height'][::-1],
                df[f'f{i + 1}.deposited'][::-1] * (1000 / kt),
                left=0,
                right=0,
            )

    # this is assuming zero error from the model
    sigma_sqr = ref_lightcurve_error**2.0

    ln_llhood = -0.5 * np.sum(
        (model_lightcurve_interped - ref_lightcurve) ** 2.0 / (sigma_sqr)
        + np.log(2 * np.pi * sigma_sqr)
    )
    if not np.isfinite(ln_llhood):
        return -np.inf

    return ln_prior + ln_llhood
