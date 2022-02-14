import numpy as np
from models.prod_functions import FarmModel
from weis.multifidelity.methods.trust_region import SimpleTrustRegion
from time import time
from os import path

np.random.seed(123)

extent = 600.
bounds = {
    "x": np.array([[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent]]),
    "y": np.array([[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent]]),
}
desvars = {
    "x": [0., 0., extent, extent, extent/2, extent/2., extent/1.25],
    "y": [0, extent, 0., extent, 0., extent, 200.],
}

wd = np.linspace(0, 360, 6)
ws = np.linspace(0, 26, 5)
model_low = FarmModel(desvars, 'jensen.pkl', input_file="jensen_input.json", wd=wd, ws=ws)

wd = np.linspace(0, 360, 18)
ws = np.linspace(0, 26, 14)
model_high = FarmModel(desvars, 'gch.pkl', input_file="gch_input.json", wd=wd, ws=ws)
trust_region = SimpleTrustRegion(
    model_low,
    model_high,
    bounds,
    disp=2,
    trust_radius=300.,
    num_initial_points=600,
)

trust_region.add_objective("AEP", scaler=-0.1)
trust_region.add_constraint("turb_spacing", upper=0.0)

trust_region.set_initial_point(model_low.flatten_desvars(desvars))

trust_region.optimize(plot=False, num_iterations=30, num_basinhop_iterations=10)

