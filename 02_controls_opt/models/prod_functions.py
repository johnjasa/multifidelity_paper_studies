from weis.multifidelity.models.base_model import BaseModel
from .mf_controls import MF_Turbine, Level2_Turbine, Level3_Turbine


class L2Turbine(BaseModel):
    
    def __init__(self, desvars_init, warmstart_file, mf_turb):
        super(L2Turbine, self).__init__(desvars_init, warmstart_file)
        dofs = ['GenDOF','TwFADOF1','PtfmPDOF']
        self.l2_turb = Level2_Turbine(mf_turb, dofs)

    def compute(self, desvars):
        outputs = self.l2_turb.compute(desvars['pc_omega'])
        print('L2 compute:', desvars['pc_omega'])
        return outputs
        
        
class L3Turbine(BaseModel):
    
    def __init__(self, desvars_init, warmstart_file, mf_turb):
        super(L3Turbine, self).__init__(desvars_init, warmstart_file)
        self.l3_turb = Level3_Turbine(mf_turb)

    def compute(self, desvars):
        outputs = self.l3_turb.compute(desvars['pc_omega'])
        print('L3 compute:', desvars['pc_omega'])
        return outputs
        
        

