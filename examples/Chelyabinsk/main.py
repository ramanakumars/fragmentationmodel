import sys
sys.path.append('../../')
import numpy as np
from fragmentationmodel.data import Data
from fragmentationmodel.my_model import FragDNest4
from fragmentationmodel.planet import Planet
import dnest4
import json

def main():
    '''
    Main function to run the DNest4 sampler.
    '''

    #Define planet and tp profile
    planet = Planet("Earth")
    planet.define_temperature_profile('../earthtp/chelyabinsk_50k.tp')

    #set base configuration
    with open('chelyabinsk.json', 'r') as infile:
        base_config = json.load(infile)

    parameters = {
        'v': {'min': 12000., 'max': 22000., 'location': 'main_body.initial_velocity'},
        'theta': {'min': 0., 'max': 45., 'location': 'main_body.initial_angle'},
        'Ch': {'min': 1e-9, 'max': 1e-8, 'location': 'main_body.ablation_coefficient', 'scale': 'log'},
        'sigma': {'min': 0.1e6, 'max': 1e6, 'location': 'main_body.initial_strength', 'scale': 'log'},
        'rho_d': {'min': 1000., 'max': 5000., 'location': 'main_body.bulk_density'},
    }   

    #load in data
    energydepdata = np.loadtxt("../energydep/ChelyabinskEnergyDep_Wheeler-et-al-2018.txt", skiprows=1)

    #Create a model instance
    model = FragDNest4(base_config, planet)

    #set parameters
    model.set_parameters(parameters)
    #set integration parameters
    model.set_integration_parameters({'dt': 0.1, 'max_time': 25, 'min_height': 0, 'max_height': 50000})
    #set light curve
    model.set_lightcurve(energydepdata[:, 0] * 1e3, energydepdata[:, 1], 0.01, 'energy_deposition') #does this need to have .get_instance()?

    #create sampler
    sampler = dnest4.DNest4Sampler(
        model,
        backend=dnest4.backends.CSVBackend(".", sep=" ")
    )

    # Run the sampler
    gen = sampler.sample(
        max_num_levels=50,
        num_steps=10000,
        new_level_interval=10000,
        num_per_step=10000,
        thread_steps=100,
        num_particles=1,
        lam=10,
        beta=100,
        seed=1234
    )

    # Do the sampling (one iteration here = one particle save)
    for i, sample in enumerate(gen):
        print(f"# Saved {i+1} particles.")
    
    # Run postprocessing
    dnest4.postprocess_abc()

    return 0

if __name__ == "__main__":
    sys.exit(main())