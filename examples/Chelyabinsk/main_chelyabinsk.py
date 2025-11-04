import sys
sys.path.append('../..')
import dnest4
import numpy as np
import multiprocessing as mp
import os
from fragmentationmodel.fragmentation_model import FragmentationModel
from fragmentationmodel.planet import Planet
from fragmentationmodel.data import Data
from fragmentationmodel.my_model import FragDNest4
import json

# Define run_chain at module level (not inside main)
def run_chain(chain_id, base_config, planet, parameters, dep_axis, lightcurve, lightcurve_err, lightcurve_type, integration_parameters):
    """Run one DNest4 chain"""
    output_dir = f"chain_{chain_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Setup model
    model = FragDNest4(base_config, planet)
    model.set_parameters(parameters)
    model.set_lightcurve(dep_axis, lightcurve, lightcurve_err, lightcurve_type)
    model.set_integration_parameters(integration_parameters)
    
    # Create sampler
    sampler = dnest4.DNest4Sampler(
        model,
        backend=dnest4.backends.CSVBackend(".", sep=" ")
    )
    
    # Run sampling
    gen = sampler.sample(
        max_num_levels=30,
        num_steps=5000,
        new_level_interval=5000,
        num_per_step=5000,
        num_particles=500,  # 25 per chain × 4 chains = 100 total
        seed=1234 + chain_id
    )
    
    for i, sample in enumerate(gen):
        print(f"Chain {chain_id}: Saved {i+1} particles")
    
    # Postprocess
    dnest4.postprocess()
    
    # Return to original directory
    os.chdir("..")
    
    return chain_id

def main():
    # Load your data and setup

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
    
    # Define filter
    kt = 4.185e12
    
    # Load in data
    energydepdata = np.loadtxt("../energydep/ChelyabinskEnergyDep_Wheeler-et-al-2018.txt", skiprows=1)
    dep_axis = energydepdata[:, 0] * 1e3
    lightcurve = energydepdata[:, 1]
    lightcurve_err = 0.01
    lightcurve_type = 'energy_deposition'
    
    integration_parameters = {'dt': 0.01, 'max_time': 25, 'min_height': 0, 'max_height': 50000}  # Increased dt!
    
    # Run multiple chains in parallel
    n_chains = 10
    
    # Create arguments for each chain
    args_list = [
        (i, base_config, planet, parameters, dep_axis, lightcurve, 
         lightcurve_err, lightcurve_type, integration_parameters)
        for i in range(n_chains)
    ]
    
    with mp.Pool(n_chains) as pool:
        results = pool.starmap(run_chain, args_list)
    
    print(f"Completed all {n_chains} chains!")
    return 0

if __name__ == "__main__":
    sys.exit(main())