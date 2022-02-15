import numpy as np
from models.prod_functions import FullCCBlade, OpenFAST
from weis.multifidelity.methods.trust_region import SimpleTrustRegion


np.random.seed(123)

bounds = {
    "blade.opt_var.twist_opt_gain": np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
    "blade.opt_var.chord_opt_gain": np.array([[0.5, 1.50], [0.5, 1.50], [0.5, 1.50]]),
}
twist_dv = {'values' : np.array([0.55, 0.55, 0.55, 0.55]),
    'idx_start' : 2}
chord_dv = {'values' : np.ones(3),
    'idx_start' : 2}

desvars = {
    "blade.opt_var.twist_opt_gain": twist_dv,
    "blade.opt_var.chord_opt_gain": chord_dv,
}
model_low = FullCCBlade(desvars, "cc_results.pkl")
model_high = OpenFAST(desvars, "of_results.pkl")
trust_region = SimpleTrustRegion(
    model_low,
    model_high,
    bounds,
    disp=True,
    trust_radius=1.0,
    num_initial_points=50,
)

trust_region.add_objective("power", scaler=-1e-6)

trust_region.optimize(plot=False)
