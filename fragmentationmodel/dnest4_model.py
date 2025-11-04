import dnest4
import numpy as np
import numpy.random as rng
from copy import deepcopy
from collections import OrderedDict
from .fragmentation_model import FragmentationModel
from .planet import Planet
from .data import Data

kt = 4.184e12

class FragDNest4():
    '''
    Framgentation model paired with DNest4.
    '''

    def __init__(self, base_config: dict, planet: Planet):
        '''
        Initialize class
        Parameter values are not stored here
        '''
        self.base_config = base_config
        self.planet = planet
        #Following values are set later with set_ functions
        self.parameter = None
        self.parameter_names = None
        self.ndims = None
        self.ref_lightcurve = None
        self.ref_dep_axis = None
        self.ref_lightcurve_error = None
        self.lightcurve_type = None
        self.integration_parameters = None

    @staticmethod
    def normalize(parameter: float, min: float, max: float, scale: str = "linear") -> float:
        '''
        Normalize the parameter to be between 0 and 1

        :param parameter: the parameter to normalize
        :param min: the minimum value of the parameter
        :param max: the maximum value of the parameter

        :returns: the normalized parameter
        '''
        if scale == "linear":
            return (parameter - min) / (max - min)
        elif scale == "log":
            return (np.log(parameter) - np.log(min)) / (np.log(max) - np.log(min))

    @staticmethod
    def denormalize(
        parameter: float, min: float, max: float, scale: str = "linear"
    ) -> float:
        '''
        Denormalize the parameter to be between min and max

        :param parameter: the normalized parameter
        :param min: the minimum value of the parameter
        :param max: the maximum value of the parameter

        :returns: the denormalized parameter
        '''
        if scale == "linear":
            return (max - min) * parameter + min
        elif scale == "log":
            return np.exp((np.log(max) - np.log(min)) * parameter + np.log(min))

    @staticmethod
    def update_dict(orig_dict: dict, new_dict: dict) -> dict:
        """
        Update the original dictionary with the new dictionary recursively
        :param orig_dict: the original dictionary
        :param new_dict: the new dictionary

        :returns: the updated dictionary
        """
        for key, value in new_dict.items():
            if isinstance(value, dict):
                orig_dict[key] = FragDNest4.update_dict(orig_dict.get(key, {}), value)
            elif isinstance(value, list):
                orig_dict[key].extend(value)
            else:
                orig_dict[key] = value
        return orig_dict

    @staticmethod
    def params_to_dict(p: np.ndarray, parameter: dict) -> dict:
        """
        Convert the parameters from the MCMC sampler to an input config format for the frag model
        :param p: the parameters from the MCMC sampler
        :param parameter: the parameters dictionary from the model (containing the locations key for each parameter)

        :returns: the updated config dictionary for the frag model
        """
        #locations: dict[str] = {key: value['location'] for key, value in parameter.items()}

        updated_config = {'main_body': {}, 'fragments': []}
        fragments = {}

        #CLAUDE
        param_keys = list(parameter.keys())

        for i, key in enumerate(param_keys):
            location = parameter[key]['location']
            value = p[i]


            # the main body is stored as `main_body.<key>` in the config
            if 'main_body' in location:
                dict_key = location.split('.')[1]
                updated_config['main_body'][dict_key] = value #FragDNest4.denormalize(
                #     p[i],
                #     parameter[key]['min'],
                #     parameter[key]['max'],
                #     scale=parameter[key].get('scale', 'linear'),
                # )

            # the fragments are stored as `fragment.<index>.<key>` in the config
            elif 'fragment' in location:
                # parse the fragment index and key from the location
                _, fragment_index, fragment_key = location.split('.')

                # first fragment is index 1 since the main body is technically index 0
                fragment_index = int(fragment_index) - 1
                if fragment_index not in fragments:
                    fragments[fragment_index] = {}
                fragments[fragment_index][fragment_key] = value #FragDNest4.denormalize(
                #     p[i],
                #     parameter[key]['min'],
                #     parameter[key]['max'],
                #     scale=parameter[key].get('scale', 'linear'),
                # )

        # sort the fragments by index
        fragments = OrderedDict(sorted(fragments.items()))

        # and then update the config with the fragments
        # in the order of their index
        for fragment in fragments.values():
            updated_config['fragments'].append(fragment)

        return updated_config

    def set_parameters(self, parameter: dict):
        '''
        Set the parameters for the model

        :param parameter: dictionary containing parameter information (min, max, location, etc.)
        '''
        self.parameter = parameter
        self.parameter_names = list(parameter.keys())
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

        #CLAUDE
        if isinstance(lightcurve_err, (int, float)):
            self.ref_lightcurve_error = lightcurve_err * self.ref_lightcurve
        else: 
            self.ref_lightcurve_error = lightcurve_err

        self.lightcurve_type = lightcurve_type

    def set_integration_parameters(self, integration_parameters: dict):
        '''
        Set the integration parameters for the model

        :param integration_parameters: the integration parameters to use for the model
        '''
        self.integration_parameters = integration_parameters

    '''
    next three functions are required for dnest4
    '''
    def from_prior(self):
        '''
        Generate parameter values from the prior.
        Returns a numpy array of parameters.
        
        NOTE: This function only takes self, no other arguments
        '''
        if self.parameter is None:
            raise ValueError("Parameters not set.")

        parameters = []
        
        for key, info in self.parameter.items():
            # width = info['max'] - info['min']
            # value = info['min'] + width * rng.rand()
            # parameters.append(value)
            if info.get('scale') == 'log':
                #log scale
                log_min = np.log(info['min'])
                log_max = np.log(info['max'])
                value = np.exp(log_min + (log_max - log_min) * rng.rand())
            else:
                #linear scale
                value = info['min'] + (info['max'] - info['min']) * rng.rand()

            parameters.append(value)

            # Debug: Print Parameter ranges
            # if not hasattr(self, '_printed_ranges'):
            #     print(f"Prior for {key}: {value:.3e} (range: [{info['min']:.3e}, {info['max']:.3e}]")

        self._printed_ranges = True
        
        return np.array(parameters, dtype=np.float64)

    def perturb(self, params):
        '''
        Perturb parameter values
        '''
        logH = 0.0
        #data_size = len(Data.get_instance().get_x())
        which = rng.randint(len(params))

        param_keys = list(self.parameter.keys())
        param_info = self.parameter[param_keys[which]]

        old_value = params[which]
        #perturb parameter
        width = param_info['max'] - param_info['min']
        params[which] += width * dnest4.randh()

        # Debug: Check perturbation size
        # if np.random.rand() < 0.01:
        #     change = abs(params[which] - old_value) / width
        #     print(f"DEBUG: Perturbed param {which} by {change*100:.1f}% of range")

        #wrap to stay in bounds 
        params[which] = dnest4.wrap(params[which], param_info['min'], param_info['max'])

        return logH


    def log_likelihood(self, params):
        '''
        Get the log-likelihood for the current set of parameters by runnin FragmentationModel and comparing the lightcurve.

        :param p: current test values of the model parameters
        :param parameter: dict containing information about each parameter (min/max values, location in config, etc.)
        :param base_config: the base configuration for the fragmentation model. The parameters in `p` will be used to update this config.
        :param planet: the planet object to use for the model (see `Planet`)
        :param integration_parameters: the integration parameters to use for the model (see `FragmentationModel.integrate`)
        :param ref_lightcurve: the reference lightcurve to compare against
        :param ref_dep_axis: the reference dependent axis (either time [s] or height [m], see `lightcurve_type`)
        :param ref_lightcurve_error: the error in the reference lightcurve (constant if equal for all time or array for each measurement, same units as `ref_lightcurve`)
        :param lightcurve_type: the type of lightcurve (either 'lightcurve' or 'energy_deposition'). If passing in a lightcurve, `ref_dep_axis` should be time (s) and `ref_lightcurve` should be in W. If passing in energy deposition, `ref_dep_axis` should be height (m) and `ref_lightcurve` should be in kt/km.

        :returns: the log-likelihood value for the current set of parameters `p`
        '''

        if not hasattr(self, '_n_calls'):
            self._n_calls = 0
            self._n_success = 0

        self._n_calls += 1

        # Debug: Print parameter values occasionally
        # if np.random.rand() < 0.01:
        #     print(f"DEBUG: params = {params[:5]}...")

        if self.ref_lightcurve is None:
            raise ValueError("Reference lightcurve not set.")
        if self.integration_parameters is None:
            raise ValueError("Integration parameters no set.")

        updated_config = self.params_to_dict(params, self.parameter)

        # DEBUG: Print what's actually in the config
        # print(f"\nDEBUG CONFIG:")
        # print(f"  velocity: {updated_config['main_body'].get('initial_velocity', 'MISSING')}")
        # print(f"  angle: {updated_config['main_body'].get('initial_angle', 'MISSING')}")
        # print(f"  Ch: {updated_config['main_body'].get('ablation_coefficient', 'MISSING')}")
        # print(f"  sigma: {updated_config['main_body'].get('initial_strength', 'MISSING')}")
        # print(f"  rho_d: {updated_config['main_body'].get('bulk_density', 'MISSING')}")


        try:
            model = FragmentationModel.load_from_dict(
                self.update_dict(deepcopy(self.base_config), updated_config), 
                self.planet
            )
        except Exception as e:
            #model creation failed (unphysical parameters)
            return -np.inf

        try: 
            df = model.integrate(**self.integration_parameters)
            self._n_success += 1
        except Exception:
            if np.random.rand() < 0.01:
                print("DEBUG: Integration failed")
            return -np.inf
            if self._n_calls % 100 == 0:
                success_rate = 100 * self._n_success / self._n_calls
                print(f"Success rate: {success_rate:.1f}% ({self._n_success}/{self._n_calls})")
            return -np.inf

        #except (ValueError, RuntimeError) as e:
            #print(f"Integration failed: {e}")
            #return -np.inf

        if self.lightcurve_type == 'lightcurve':
            # mask = np.isfinite(df["main.total"])
            lightcurve = df['main.total'].copy()
            for i in range(len(model.fragments)):
                lightcurve += df[f'f{i + 1}.total']

            # shift the model lightcurve to match the reference lightcurve by aligning the lightcurve peak
            t_peak_model = df['main.time'][np.argmax(lightcurve)]
            t_peak_ref = self.ref_dep_axis[np.argmax(self.ref_lightcurve)]

            model_lightcurve_interped = np.interp(
                self.ref_dep_axis - t_peak_ref,
                df['main.time'] - t_peak_model,
                lightcurve,
                left=0,
                right=0,
            )

            # Debug: Check model output
            # print(f"\n=== DETAILED DEBUG ===")
            # print(f"Params: {params}")
            # print(f"Model lightcurve stats:")
            # print(f"  Min: {np.min(model_lightcurve_interped):.3e}")
            # print(f"  Max: {np.max(model_lightcurve_interped):.3e}")
            # print(f"  Mean: {np.mean(model_lightcurve_interped):.3e}")
            # print(f"  Sum: {np.sum(model_lightcurve_interped):.3e}")
            # print(f"Data lightcurve stats:")
            # print(f"  Min: {np.min(self.ref_lightcurve):.3e}")
            # print(f"  Max: {np.max(self.ref_lightcurve):.3e}")
            # print(f"  Mean: {np.mean(self.ref_lightcurve):.3e}")

        elif self.lightcurve_type == 'energy_deposition':
            model_lightcurve_interped = np.interp(
                self.ref_dep_axis,
                df['main.height'][::-1],
                df['main.deposited'][::-1] * (1000 / kt),
                left=0,
                right=0,
            )
            for i in range(len(model.fragments)):
                model_lightcurve_interped += np.interp(
                    self.ref_dep_axis,
                    df[f'f{i + 1}.height'][::-1],
                    df[f'f{i + 1}.deposited'][::-1] * (1000 / kt),
                    left=0,
                    right=0,
                )


        sigma_sqr = self.ref_lightcurve_error**2.0

        mask = sigma_sqr > 0

        if not np.any(mask):
            return -np.inf

        n_data = np.sum(mask)
        residuals =  model_lightcurve_interped[mask] - self.ref_lightcurve[mask]
        chi_squared = np.sum((residuals / self.ref_lightcurve_error[mask])**2)

        # if np.random.rand() < 0.05:
        #     print(f"DEBUG: Model range: [{np.min(model_lightcurve_interped):.2e}, " 
        #           f"{np.max(model_lightcurve_interped):.2e}]")
        #     print(f"DEBUG: Data range; [{np.min(self.ref_lightcurve):.2e}, "
        #           f"{np.max(self.ref_lightcurve):.2e}]")
        #     print(f"DEBUG: Mean residual: {np.mean(residuals):.2e}")

        # print(f"Residual stats:")
        # print(f"  Mean: {np.mean(residuals):.3e}")
        # print(f"  RMS: {np.sqrt(np.mean(residuals**2)):.3e}")
        # print(f"  Max: {np.max(np.abs(residuals)):.3e}")


        ln_llhood = (
            -0.5 * n_data * np.log(2 * np.pi * np.mean(sigma_sqr[mask]))
            - 0.5 * chi_squared
        )

        if not np.isfinite(ln_llhood):
            return -np.inf

        #if np.random.rand() < 0.05:
            #print(f"DEBUG: log_likelihood = {ln_llhood}")
            #print(f"DEBUG: chi_squared = {chi_squared:.2f}, logL = {ln_llhood:.2f}")

        # print(f"Chi-squared: {chi_squared:.3e}")
        # print(f"Log-likelihood: {ln_llhood:.3e}")
        # print("=" * 40)

        return ln_llhood
