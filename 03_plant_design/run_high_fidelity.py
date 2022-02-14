import numpy as np
from models.prod_functions import FarmModel
from time import time
from os import path
import openmdao.api as om

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

extent = 600.
bounds = {
    "x": np.array([[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent]]),
    "y": np.array([[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent],[0.0, extent]]),
}
desvars = {
    "x": [0., 0., extent, extent, extent/2, extent/2., extent/1.25],
    "y": [0, extent, 0., extent, 0., extent, 200.],
}

class Farm(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('desvars')
        
    def setup(self):
        desvars = self.options["desvars"]
        for key in desvars:
            self.add_input(key, val=desvars[key])
        
        self.add_output('AEP', val=0., units='MW*h') # TODO: check units
        self.add_output('turb_spacing', val=0.) # TODO: check units
        
        wd = np.linspace(0, 360, 18)
        ws = np.linspace(0, 26, 14)
        
        self.model = FarmModel(desvars, 'gch.pkl', input_file="gch_input.json", wd=wd, ws=ws)

    def compute(self, inputs, outputs):
        desvars = self.options["desvars"]
        model_outputs = self.model.compute(inputs)
        outputs['AEP'] = model_outputs['AEP']
        outputs['turb_spacing'] = model_outputs['turb_spacing']


p = om.Problem(model=om.Group(num_par_fd=28))
model = p.model
model.approx_totals(method='fd', step=1e-6, form='central')
comp = model.add_subsystem('Farm', Farm(desvars=desvars), promotes=['*'])

for key in desvars:
    model.set_input_defaults(key, val=desvars[key])
    
s = time()

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = "SNOPT"

for key in desvars:
    model.add_design_var(key, lower=bounds[key][:, 0], upper=bounds[key][:, 1])
model.add_constraint('turb_spacing', upper=0.)
model.add_objective('AEP', ref=-10.)

p.driver.recording_options['includes'] = ['*']
p.driver.recording_options['record_objectives'] = True
p.driver.recording_options['record_constraints'] = True
p.driver.recording_options['record_desvars'] = True
p.driver.recording_options['record_inputs'] = True
p.driver.recording_options['record_outputs'] = True
p.driver.recording_options['record_residuals'] = True

p.driver.options['debug_print'] = ['desvars','ln_cons','nl_cons','objs','totals']

recorder = om.SqliteRecorder("cases.sql")
p.driver.add_recorder(recorder)

p.setup(mode='fwd')
p.run_driver()


# import floris.tools as wfct
# import matplotlib.pyplot as plt
# 
# # Get horizontal plane at default height (hub-height)
# hor_plane = p.model.Farm.model.fi.get_hor_plane()
# 
# # Plot and show
# fig, ax = plt.subplots()
# wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
# plt.show()
# 
