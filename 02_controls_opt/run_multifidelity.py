import numpy as np
from weis.multifidelity.methods.trust_region import SimpleTrustRegion
from models.prod_functions import L2Turbine, L3Turbine
from models.mf_controls import MF_Turbine, Level2_Turbine, Level3_Turbine


bounds = {'pc_omega' : np.array([[0.10, 0.4]])}
desvars = {'pc_omega' : np.array([0.22])}
mf_turb = MF_Turbine()
mf_turb.n_cores = 1
model_low = L2Turbine(desvars, 'L2Turbine.pkl', mf_turb)
model_high = L3Turbine(desvars, 'L3Turbine.pkl', mf_turb)

np.random.seed(123)

trust_region = SimpleTrustRegion(
    model_low,
    model_high,
    bounds,
    disp=2,
    trust_radius=0.5,
    num_initial_points=2,
)

trust_region.add_objective("TwrBsMyt_DEL", scaler=1e-5)
trust_region.add_constraint("GenSpeed_Max", upper=9.)
trust_region.add_constraint("PtfmPitch_Max", upper=5.)

trust_region.optimize(plot=False, num_basinhop_iterations=5)
