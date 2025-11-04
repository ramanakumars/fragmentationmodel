import sys
sys.path.append('../..')
import time
import json
import numpy as np
from fragmentationmodel.my_model import FragDNest4
from fragmentationmodel.planet import Planet

planet = Planet("Earth")
planet.define_temperature_profile('../earthtp/chelyabinsk_50k.tp')

with open('chelyabinsk.json', 'r') as infile:
        base_config = json.load(infile)

parameters = {
        'v': {'min': 12000., 'max': 22000., 'location': 'main_body.initial_velocity'},
        'theta': {'min': 0., 'max': 45., 'location': 'main_body.initial_angle'},
        'Ch': {'min': 1e-9, 'max': 1e-8, 'location': 'main_body.ablation_coefficient', 'scale': 'log'},
        'sigma': {'min': 0.1e6, 'max': 1e6, 'location': 'main_body.initial_strength', 'scale': 'log'},
        'rho_d': {'min': 1000., 'max': 5000., 'location': 'main_body.bulk_density'},
    }

energydepdata = np.loadtxt("../energydep/ChelyabinskEnergyDep_Wheeler-et-al-2018.txt", skiprows=1)
dep_axis = energydepdata[:, 0] * 1e3
lightcurve = energydepdata[:, 1]
lightcurve_err = 0.01
lightcurve_type = 'energy_deposition'

# Setup
model = FragDNest4(base_config, planet)
model.set_parameters(parameters)
model.set_lightcurve(dep_axis, lightcurve, lightcurve_err, lightcurve_type)

# Test one set of parameters with different dt values
test_params = model.from_prior()

print("Testing different timesteps:\n")
for dt in [0.005, 0.01, 0.05, 0.1, 0.2]:
    model.set_integration_parameters({'dt': dt, 'max_time': 25, 'min_height': 0, 'max_height': 50000})
    
    start = time.time()
    try:
        logL = model.log_likelihood(test_params)
        elapsed = time.time() - start
        print(f"dt={dt:5.3f}: logL={logL:12.2e}, time={elapsed:6.2f}s")
    except Exception as e:
        print(f"dt={dt:5.3f}: FAILED - {e}")