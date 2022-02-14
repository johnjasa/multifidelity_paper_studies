from weis.multifidelity.models.base_model import BaseModel
import floris.tools as wfct
import os
import numpy as np
import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.cut_plane as cp
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.yaw_wind_rose import YawOptimizationWindRose
from scipy.spatial.distance import cdist


class FarmModel(BaseModel):
    
    def __init__(self, desvars_init, warmstart_file, input_file, calls_between_saving=1, wd=np.arange(0, 360, 60), ws=np.arange(0, 26, 2)):
        super(FarmModel, self).__init__(desvars_init, warmstart_file)
        # Initialize the FLORIS interface fi
        this_directory = os.path.abspath(os.path.dirname(__file__))
        input_file = os.path.join(this_directory, input_file)
        self.fi = wfct.floris_interface.FlorisInterface(input_file)
        
        if 'gch' in input_file:
            self.fi.set_gch(True)
            self.type = "gch"
        else:
            self.type = "jensen"
        
        wind_rose = rose.WindRose()

        self.df = wind_rose.make_wind_rose_from_weibull(wd=wd, ws=ws)
        
        # Below minimum wind speed, assumes power is zero.
        minimum_ws = 3.0
        
        self.yaw_opt = YawOptimizationWindRose(self.fi, self.df.wd, self.df.ws, minimum_ws=minimum_ws)

    def compute(self, desvars):
        x = desvars['x']
        y = desvars['y']
        
        # Set to 2x2 farm
        self.fi.reinitialize_flow_field(layout_array=[list(x), list(y)])
        
        # Calculate wake
        self.fi.calculate_wake()
        
        # Determine baseline power with and without wakes
        df_base = self.yaw_opt.calc_baseline_power()

        # Initialize power rose
        power_rose = pr.PowerRose()
        power_rose.make_power_rose_from_user_data(
            'wind farm', self.df, df_base["power_no_wake"], df_base["power_baseline"]
        )
        
        outputs = {}
        outputs['AEP'] = power_rose.total_baseline
        
        rho = 500
        min_dist = 252
                
        # Sped up distance calc here using vectorization
        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)
                
        g = 1 - np.array(dist) / min_dist
        
        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)
        
        outputs['turb_spacing'] = KS_constraint[0][0]
        
        print('Computing AEP for ', self.type, outputs['AEP'])
        
        return outputs