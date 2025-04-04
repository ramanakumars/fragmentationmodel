import emcee
import numpy as np
from multiprocessing import Pool

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


class MCMCSolver:
    def __init__(self):
        pass

    def set_parameters(self, parameters: dict):
        self.parameters = parameters
        self.parameter_names = list(parameters.keys())
        self.ndims = len(self.parameter_names)

    def set_lightcurve(self, dependent_axis: np.ndarray, lightcurve: np.ndarray, lightcurve_err: np.ndarray | float, lightcurve_type: str = 'lightcurve'):
        '''
        Set the observe lightcurve for fitting

        :param dependent_axis: the input x-dimension (either time [s] or height [m], see `lightcurve_type`)
        :param lightcurve: the input lightcurve (either W or kt/km, see `lightcurve_type`)
        :param lightcurve_err: error in lightcurve (constant if equal for all time or array for each measurement, same units as `lightcurve`)
        :param lightcurve_type: the type of lightcurve (either 'lightcurve' or 'energy_deposition'). If passing in a lightcurve, `dependent_axis` should be time (s) and `lightcurve` should be in W. If passing in energy deposition, `dependent_axis` should be height (m) and `lightcurve` should be in W/km.

        :raises AssertionError: if input spectrum wavelength range is outside the model config wavelength range
        '''
        if lightcurve_type.lower() not in ['lightcurve', 'energy_deposition']:
            raise ValueError("lightcurve_type must be either 'lightcurve' or 'energy_deposition'")

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

    def run_mcmc(self, get_model: callable, num_steps: int = 150, n_walkers: int = 10, verbose: bool = True, threads: int = 1):
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
            self.sampler = emcee.EnsembleSampler(n_walkers, self.ndims, log_likelihood,
                                                    args=(get_model, self.parameters, self.integration_parameters, self.ref_dep_axis,
                                                    self.ref_lightcurve_error, self.ref_lightcurve, self.lightcurve_type),
                                                    parameter_names=self.parameter_names, pool=pool)
            return self.sampler.run_mcmc(initial_guess, num_steps, progress=verbose)


def likelihood_prior(p: dict[float], parameters: dict) -> float:
    '''
    Get the model prior for each parameters.
    This is essentially a wrapper function to set a prior of -infinity when the parameter is outside its defined range

    :param p: list containing current values of the parameters
    :param parameters: dict containing min/max values for each parameter

    :returns: the log-probability for the prior
    '''
    check = [(param > 0) & (param < 1) for key, param in p.items()]
    if np.all(check):
        return 0.
    else:
        return -np.inf


def log_likelihood(p: dict[float], get_model: callable, parameters: dict, integration_parameters: dict,
                   ref_dep_axis: np.ndarray, ref_lightcurve_error: np.ndarray, ref_lightcurve: np.ndarray, lightcurve_type: str) -> float:
    '''
    Get the log-likelihood for the current set of parameters by running FragmentationModel and comparing the lightcurve.
    This is what emcee will use to find the global minima. Returns -inf if any parameter in `p` is out of range.

    :param p: current values of the model parameters
    :param get_model: function that returns the new `FragmentationModel` object
    :param integration_parameters: the integration parameters to use for the model
    :param ref_dep_axis: the dependent axis for the reference lightcurve (either time [s] or height [m])
    :param ref_lightcurve_error: the error for the reference lightcurve for each time/height
    :param ref_lightcurve: the reference lightcurve for each time/height
    :param lightcurve_type: the type of lightcurve (either 'lightcurve' or 'energy_deposition'). If passing in a lightcurve, `dependent_axis` should be time (s) and `lightcurve` should be in W. If passing in energy deposition, `dependent_axis` should be height (m) and `lightcurve` should be in W/km.

    :returns: the log-likelihood value for the current set of parameters `p`
    '''
    ln_prior = likelihood_prior(p, parameters)

    if not np.isfinite(ln_prior):
        return -np.inf

    de_normalized_p = {key: denormalize(param, parameters[key]['min'], parameters[key]['max']) for key, param in p.items()}

    model = get_model(**de_normalized_p)

    df = model.integrate(**integration_parameters)

    if lightcurve_type == 'lightcurve':
        lightcurve = df['main.lightcurve']
        for i in range(len(model.fragments)):
            lightcurve += df[f'f{i + 1}.lightcurve']

        model_lightcurve_interped = np.interp(ref_dep_axis, df['main.time'], lightcurve)
    elif lightcurve_type == 'energy_deposition':
        model_lightcurve_interped = np.interp(ref_dep_axis, df['main.height'], df['main.deposited'] * (1000 / kt), left=0, right=0)
        for i in range(len(model.fragments)):
            model_lightcurve_interped += np.interp(ref_dep_axis, df[f'f{i + 1}.height'], df[f'f{i + 1}.deposited'] * (1000 / kt), left=0, right=0)

    # this is assuming zero error from the model
    sigma_sqr = ref_lightcurve_error ** 2.

    ln_llhood = -0.5 * np.sum((model_lightcurve_interped - ref_lightcurve)**2. / (sigma_sqr) + np.log(2 * np.pi * sigma_sqr))
    if not np.isfinite(ln_llhood):
        return -np.inf

    return ln_prior + ln_llhood
