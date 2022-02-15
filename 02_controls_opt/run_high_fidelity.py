import numpy as np
from models.prod_functions import L3Turbine
from models.mf_controls import MF_Turbine
from time import time
from os import path
import openmdao.api as om


bounds = np.array([0.10, 0.4])
desvars = {'pc_omega' : np.array([0.22])}

class Model(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('desvars')
        
    def setup(self):
        desvars = self.options["desvars"]
        for key in desvars:
            self.add_input(key, val=desvars[key])
        
        self.add_output('TwrBsMyt_DEL', val=0.)
        self.add_output('GenSpeed_Max', val=0.)
        self.add_output('GenSpeed_Std', val=0.)
        self.add_output('PtfmPitch_Max', val=0.)
        self.add_output('PtfmPitch_Std', val=0.)
        
        mf_turb = MF_Turbine()
        mf_turb.n_cores = 1
        
        self.model = L3Turbine(desvars, 'L3Turbine.pkl', mf_turb)

    def compute(self, inputs, outputs):
        desvars = self.options["desvars"]
        model_outputs = self.model.compute(inputs)
        outputs['TwrBsMyt_DEL'] = model_outputs['TwrBsMyt_DEL']
        outputs['GenSpeed_Max'] = model_outputs['GenSpeed_Max']
        outputs['GenSpeed_Std'] = model_outputs['GenSpeed_Std']
        outputs['PtfmPitch_Max'] = model_outputs['PtfmPitch_Max']
        outputs['PtfmPitch_Std'] = model_outputs['PtfmPitch_Std']
        
        
        
p = om.Problem(model=om.Group())
model = p.model
model.approx_totals(method='fd', step=1e-3, form='central')
comp = model.add_subsystem('Model', Model(desvars=desvars), promotes=['*'])

for key in desvars:
    model.set_input_defaults(key, val=desvars[key])
    
s = time()

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = "SNOPT"

for key in desvars:
    model.add_design_var(key, lower=bounds[0], upper=bounds[1])
model.add_constraint('GenSpeed_Max', upper=9.)
model.add_constraint('PtfmPitch_Max', upper=5.)
model.add_objective('TwrBsMyt_DEL', ref=1.e5)

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
