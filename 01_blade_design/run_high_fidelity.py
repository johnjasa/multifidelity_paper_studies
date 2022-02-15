import numpy as np
from models.prod_functions import OpenFAST
from scipy.optimize import minimize
from time import time
from os import path
import openmdao.api as om

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

class OF(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('desvars')
        
    def setup(self):
        desvars = self.options["desvars"]
        for key in desvars:
            trimmed_key = key.split('.')[-1]
            self.add_input(trimmed_key, val=desvars[key]['values'])
        
        self.add_output('power', val=0., units='MW')
        self.model = OpenFAST(desvars, 'OF.pkl')

    def compute(self, inputs, outputs):
        desvars = self.options["desvars"]
        
        model_inputs = {}
        for key in desvars:
            trimmed_key = key.split('.')[-1]
            model_inputs[key] = inputs[trimmed_key]
            
        model_outputs = self.model.compute(model_inputs)
        outputs['power'] = model_outputs['power']


p = om.Problem(model=om.Group(num_par_fd=7))
model = p.model
model.approx_totals(method='fd')
comp = model.add_subsystem('OF', OF(desvars=desvars), promotes=['*'])

for key in desvars:
    trimmed_key = key.split('.')[-1]
    model.set_input_defaults(trimmed_key, val=desvars[key]['values'])
    
s = time()

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = "SNOPT"

for key in desvars:
    trimmed_key = key.split('.')[-1]
    model.add_design_var(trimmed_key, lower=bounds[key][:, 0], upper=bounds[key][:, 1])
model.add_objective('power', ref=-1.e4)

p.driver.recording_options['includes'] = ['*']
p.driver.recording_options['record_objectives'] = True
p.driver.recording_options['record_constraints'] = True
p.driver.recording_options['record_desvars'] = True
p.driver.recording_options['record_inputs'] = True
p.driver.recording_options['record_outputs'] = True
p.driver.recording_options['record_residuals'] = True

recorder = om.SqliteRecorder("cases.sql")
p.driver.add_recorder(recorder)

p.setup(mode='fwd')
p.run_driver()
