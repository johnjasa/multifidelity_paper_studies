'''
Functions for running linear and nonlinear control parameter optimizations

- Run full set of DLCs
- Process and find Worst Case
- Nonlinear
    - Tune ROSCO, update
    - Run single, worst case DLC
- Linear  (currently: only doing this!)
    - Generate linear model from nonlinear simulation
    - Tune linear ROSCO
    - Run linear simulation
- Process DEL, other measures for cost function

'''
from weis.aeroelasticse.runFAST_pywrapper import runFAST_pywrapper_batch
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from weis.aeroelasticse.CaseGen_IEC import CaseGen_IEC
from weis.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
from weis.aeroelasticse.Util.FileTools import save_yaml, load_yaml
from weis.aeroelasticse.LinearFAST import LinearFAST
from weis.aeroelasticse.FAST_post   import FAST_IO_timeseries
from pCrunch.Analysis import Loads_Analysis
from weis.aeroelasticse.Turbsim_mdao.turbsim_file import TurbSimFile



# weis control modules
import weis.control.LinearModel as lin_mod


# pCrunch Modules and instantiation
import matplotlib.pyplot as plt 

from ROSCO_toolbox import utilities as ROSCO_utilities
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine

# WISDEM modules
from weis.aeroelasticse.Util import FileTools

# Batch Analysis
from pCrunch import pdTools
from pCrunch import Processing, Analysis


import numpy as np
import sys, os, platform, yaml

class MF_Turbine(object):
    '''
    Multifidelity turbine object:
    - Level 2 linear openfast model
    - Level 3 full nonlinear openfast simulation

    Both models use the same wind inputs, via case_inputs, iec attributes

    '''

    def __init__(self,level2_wind_speeds=[14,16,18],level3_wind_speeds=[16]):
        # Turbine Model
        # Select Turbine Model
        import weis
        weis_dir = os.path.dirname(os.path.dirname(os.path.abspath(weis.__file__)))
        model_dir               = os.path.join(weis_dir, 'examples/01_aeroelasticse/OpenFAST_models')
        output_dir                = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        self.FAST_directory     = os.path.join(model_dir, 'IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi')   # Path to fst directory files
        self.FAST_InputFile     = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)

        # level 2 and 3 directories
        self.FAST_level2_directory  = os.path.join(output_dir,'outputs','iea_semi','level2')
        self.FAST_level3_directory  = os.path.join(output_dir,'outputs','iea_semi','level3')

        # Parallel Processing
        self.n_cores            = 1

        # Set up common controller
        # Load controller from yaml file 
        parameter_filename  = os.path.join(weis_dir,'ROSCO_toolbox/Tune_Cases/IEA15MW.yaml')
        inps = yaml.safe_load(open(parameter_filename))
        path_params         = inps['path_params']
        turbine_params      = inps['turbine_params']
        controller_params   = inps['controller_params']

        controller_params['omega_pc'] = 0.15

        turbine         = ROSCO_turbine.Turbine(turbine_params)
        controller      = ROSCO_controller.Controller(controller_params)

        turbine.load_from_fast(self.FAST_InputFile,
            self.FAST_directory, \
                dev_branch=True)

        controller.tune_controller(turbine)

        self.turbine        = turbine
        controller.turbine  = turbine
        self.controller     = controller

        # Set up cases
        # Turbine inputs
        iec = CaseGen_IEC()
        iec.overwrite           = False
        iec.Turbine_Class       = 'I'   # Wind class I, II, III, IV
        iec.Turbulence_Class    = 'B'   # Turbulence class 'A', 'B', or 'C'
        iec.D                   = 240.  # Rotor diameter to size the wind grid
        iec.z_hub               = 150.  # Hub height to size the wind grid
        cut_in                  = 4.    # Cut in wind speed
        cut_out                 = 25.   # Cut out wind speed
        n_ws                    = 3    # Number of wind speed bins
        TMax                    = 800.    # Length of wind grids and OpenFAST simulations, suggested 720 s
        Vrated                  = 10.59 # Rated wind speed
        Ttrans                  = max([0., TMax - 400.])  # Start of the transient for DLC with a transient, e.g. DLC 1.4
        TStart                  = 0 # Start of the recording of the channels of OpenFAST

        # Initial conditions to start the OpenFAST runs
        u_ref     = np.arange(3.,26.) # Wind speed
        pitch_ref = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5058525323662666, 5.253759185225932, 7.50413344606208, 9.310153958810268, 10.8972969450052, 12.412247669440042, 13.883219268525659, 15.252012626933068, 16.53735488246438, 17.76456777500061, 18.953261878035104, 20.11055307762722, 21.238680277668898, 22.30705111326602, 23.455462501156205] # Pitch values in deg
        omega_ref = [2.019140272160114, 2.8047214918577925, 3.594541645994511, 4.359025795823625, 5.1123509774611025, 5.855691196288371, 6.589281196735111, 7.312788026081227, 7.514186181824161, 7.54665511646938, 7.573823812448151, 7.600476033113538, 7.630243938880304, 7.638301051122195, 7.622050377183605, 7.612285710588359, 7.60743945212863, 7.605865650155881, 7.605792924227456, 7.6062185247519825, 7.607153933765292, 7.613179734210654, 7.606737845170748] # Rotor speeds in rpm
        iec.init_cond = {}
        iec.init_cond[("ElastoDyn","RotSpeed")]        = {'U':u_ref}
        iec.init_cond[("ElastoDyn","RotSpeed")]['val'] = omega_ref
        iec.init_cond[("ElastoDyn","BlPitch1")]        = {'U':u_ref}
        iec.init_cond[("ElastoDyn","BlPitch1")]['val'] = pitch_ref
        iec.init_cond[("ElastoDyn","BlPitch2")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
        iec.init_cond[("ElastoDyn","BlPitch3")]        = iec.init_cond[("ElastoDyn","BlPitch1")]
        iec.init_cond[("HydroDyn","WaveHs")]           = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
        iec.init_cond[("HydroDyn","WaveHs")]['val']    = [1.101917033, 1.101917033, 1.179052649, 1.315715154, 1.536867124, 1.835816514, 2.187994638, 2.598127096, 3.061304068, 3.617035443, 4.027470219, 4.51580671, 4.51580671, 6.98, 10.7]
        iec.init_cond[("HydroDyn","WaveTp")]           = {'U':[3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 40, 50]}
        iec.init_cond[("HydroDyn","WaveTp")]['val']    = [8.515382435, 8.515382435, 8.310063688, 8.006300889, 7.6514231, 7.440581338, 7.460834063, 7.643300307, 8.046899942, 8.521314105, 8.987021024, 9.451641026, 9.451641026, 11.7, 14.2]
        iec.init_cond[("HydroDyn","PtfmSurge")]        = {'U':[3., 15., 25.]}
        iec.init_cond[("HydroDyn","PtfmSurge")]['val'] = [4., 15., 10.]
        iec.init_cond[("HydroDyn","PtfmPitch")]        = {'U':[3., 15., 25.]}
        iec.init_cond[("HydroDyn","PtfmPitch")]['val'] = [-1., 3., 1.3]
        iec.init_cond[("HydroDyn","PtfmHeave")]        = {'U':[3., 25.]}
        iec.init_cond[("HydroDyn","PtfmHeave")]['val'] = [0.5,0.5]

        # DLC inputs
        iec.dlc_inputs = {}

        if not isinstance(level3_wind_speeds,list):
            level3_wind_speeds = [level3_wind_speeds]

        iec.dlc_inputs['DLC']   = [1.1]
        iec.dlc_inputs['Seeds'] = [[25]]
        iec.dlc_inputs['U']     = [level3_wind_speeds]
        
        iec.dlc_inputs['Yaw']   = [[]]
        iec.PC_MaxRat           = 2.
        iec.uniqueSeeds         = True
        iec.uniqueWaveSeeds     = True

        iec.TStart              = Ttrans
        iec.TMax                = TMax    # wind file length
        iec.transient_dir_change        = 'both'  # '+','-','both': sign for transient events in EDC, EWS
        iec.transient_shear_orientation = 'both'  # 'v','h','both': vertical or horizontal shear for EWS


        # Naming, file management, etc
        iec.wind_dir        = os.path.join(weis_dir,'wind/IEA-15MW')
        iec.case_name_base  = 'level3'

        if self.n_cores > 1:
            iec.parallel_windfile_gen = True
        else:
            iec.parallel_windfile_gen = False
        iec.run_dir = self.FAST_level3_directory

        # Run case generator / wind file writing
        case_inputs = {}
        case_inputs[("Fst","TMax")]              = {'vals':[TMax], 'group':0}
        case_inputs[("Fst","TStart")]            = {'vals':[TStart], 'group':0}
        case_inputs[("Fst","OutFileFmt")]        = {'vals':[2], 'group':0}

        case_inputs[("ServoDyn","PCMode")]       = {'vals':[5], 'group':0}
        case_inputs[("ServoDyn","VSContrl")]     = {'vals':[5], 'group':0}

        # Stop Generator from Turning Off
        case_inputs[('ServoDyn', 'GenTiStr')] = {'vals': ['True'], 'group': 0}
        case_inputs[('ServoDyn', 'GenTiStp')] = {'vals': ['True'], 'group': 0}
        case_inputs[('ServoDyn', 'SpdGenOn')] = {'vals': [0.], 'group': 0}
        case_inputs[('ServoDyn', 'TimGenOn')] = {'vals': [0.], 'group': 0}
        case_inputs[('ServoDyn', 'GenModel')] = {'vals': [1], 'group': 0}

        # Set control parameters
        discon_vt = ROSCO_utilities.DISCON_dict(turbine,controller)
        for discon_input in discon_vt:
            case_inputs[('DISCON_in',discon_input)] = {'vals': [discon_vt[discon_input]], 'group': 0}


        if platform.system() == 'Windows':
            path2dll = os.path.join(weis_dir, 'local/lib/libdiscon.dll')
        elif platform.system() == 'Darwin':
            path2dll = os.path.join(weis_dir, 'local/lib/libdiscon.dylib')
        else:
            path2dll = os.path.join(weis_dir, 'local/lib/libdiscon.so')
        case_inputs[("ServoDyn","DLL_FileName")] = {'vals':[path2dll], 'group':0}


        channels = {}
        for var in ["TipDxc1", "TipDyc1", "TipDzc1", "TipDxb1", "TipDyb1", "TipDxc2", "TipDyc2", \
            "TipDzc2", "TipDxb2", "TipDyb2", "TipDxc3", "TipDyc3", "TipDzc3", "TipDxb3", "TipDyb3", \
                "RootMxc1", "RootMyc1", "RootMzc1", "RootMxb1", "RootMyb1", "RootMxc2", "RootMyc2", \
                    "RootMzc2", "RootMxb2", "RootMyb2", "RootMxc3", "RootMyc3", "RootMzc3", "RootMxb3",\
                        "RootMyb3", "TwrBsMxt", "TwrBsMyt", "TwrBsMzt", "GenPwr", "GenTq", "RotThrust",\
                            "RtAeroCp", "RtAeroCt", "RotSpeed", "BldPitch1", "TTDspSS", "TTDspFA", \
                                "NcIMUTAxs", "NcIMUTAys", "NcIMUTAzs", "NcIMURAxs", "NcIMURAys", "NcIMURAzs", \
                                    "NacYaw", "Wind1VelX", "Wind1VelY", "Wind1VelZ", "LSSTipMxa","LSSTipMya",\
                                    "LSSTipMza","LSSTipMxs","LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs", \
                                        "TipRDxr", "TipRDyr", "TipRDzr"]:
            channels[var] = True
        
        
        
        # Generate cases and safe to MF_Turbine object
        self.case_list, self.case_name_list, self.dlc_list = iec.execute(case_inputs=case_inputs)
        self.case_inputs = case_inputs

        # Save shared info to MF_Turbine
        self.iec                    = iec
        self.channels               = channels
        self.level3_wind_speeds     = level3_wind_speeds
        self.level2_wind_speeds     = level2_wind_speeds

    def compare(self,dofs,overwrite=True):
        ''' 
        Compare level 2 and 3 timeseries, for debugging purposes

        '''
        self.gen_level2_model(dofs, overwrite)

        # Extract disturbance
        dist = []
        for case in self.case_list:
            ts_file     = TurbSimFile(case[('InflowWind','FileName_BTS')])
            ts_file.compute_rot_avg(self.iec.D/2)
            u_h         = ts_file['rot_avg'][0,:]
            tt          = ts_file['t']
            dist.append({'Time':tt, 'Wind': u_h})

        # Run level 2
        self.run_level2(self.controller,dist)

        # Run level 3
        self.run_level3(self.controller, overwrite)

        # comparison plot, used to be in Level 2, not sure if information is here
        if True:
            comp_channels = ['RtVAvgxh','GenSpeed','BldPitch1','TwrBsMyt','PtfmPitch']
            fig = [None] * len(self.level3_out)
            ax = [None] * len(comp_channels)
            
            for iFig, (l2_out, l3_out) in enumerate(zip(self.level2_out,self.level3_out)):
                fig[iFig] = plt.figure()

                for iPlot, chan in enumerate(comp_channels):
                    ax[iPlot] = plt.subplot(len(comp_channels),1,iPlot+1)
                    # level 3 output
                    try:
                        ax[iPlot].plot(l3_out['Time'],l3_out[chan])
                    except:
                        print(chan + ' is not in OpenFAST OutList')

                    # level 2 output
                    try:
                        ax[iPlot].plot(l2_out['Time'],l2_out[chan])
                    except:
                        print(chan + ' is not in Linearization OutList')
                    ax[iPlot].set_ylabel(chan)
                    ax[iPlot].grid(True)
                    if not iPlot == (len(comp_channels) - 1):
                        ax[iPlot].set_xticklabels([])
                        
                ax[iPlot].set_xlabel('Time, seconds')

                fig[iFig].legend(('Level 3','Level 2'),ncol=2,loc=9)

            plt.savefig('L2_L3.png', bbox_inches='tight', dpi=450)
            
            return fig, ax


    def gen_level2_model(self,dofs,overwrite=True):
        '''
            dofs: list of strings representing ElastoDyn DOFs that will be linearized, including:
                    - FlapDOF1, FlapDOF2, EdgeDOF, TeetDOF, DrTrDOF, GenDOF, YawDOF
                    - TwFADOF1, TwFADOF2, TwSSDOF1, TwSSDOF2,
                    - PtfmSgDOF, PtfmSwDOF, PtfmHvDOF, PtfmRDOF, PtfmPDOF, PtfmYDOF
        '''
        lin_fast = LinearFAST(FAST_ver='OpenFAST', dev_branch=True)

        # fast info
        lin_fast.weis_dir                 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep
        
        lin_fast.FAST_InputFile           = self.FAST_InputFile   # FAST input file (ext=.fst)
        lin_fast.FAST_directory           = self.FAST_directory
        lin_fast.FAST_runDirectory        = self.FAST_level2_directory
        lin_fast.FAST_linearDirectory     = self.FAST_level2_directory
        lin_fast.debug_level              = 2
        lin_fast.dev_branch               = True
        lin_fast.write_yaml               = True
        
        lin_fast.v_rated                    = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
        lin_fast.wind_speeds                 = self.level2_wind_speeds
        lin_fast.DOFs                       = dofs  # enable with 
        lin_fast.TMax                       = 1600   # should be 1000-2000 sec or more with hydrodynamic states
        lin_fast.NLinTimes                  = 12

        lin_fast.overwrite_outfiles       = overwrite

        # simulation setup
        lin_fast.cores                      = self.n_cores

        # overwrite steady & linearizations
        lin_fast.overwrite        = overwrite           # for debugging only
        
        # run OpenFAST linearizations
        lin_fast.gen_linear_model()

        self.LinearTurbine = lin_mod.LinearTurbineModel(
            self.FAST_level2_directory,
            lin_fast.case_name_list,
            lin_fast.NLinTimes,
            )

    def run_level2(self,controller,disturbance):
        controller.tune_controller(self.turbine)
        linCont             = lin_mod.LinearControlModel(controller)
        self.level2_out     = []
        for dist in disturbance:
            l2_out, _, P_cl = self.LinearTurbine.solve(dist,Plot=False,open_loop=False,controller=linCont)
            self.level2_out.append(l2_out)


    def run_level3(self,controller,overwrite=True):
        controller.tune_controller(self.turbine)
        # Run FAST cases
        fastBatch                   = runFAST_pywrapper_batch(FAST_ver='OpenFAST',dev_branch = True)
        
        # Select Turbine Model
        fastBatch.FAST_directory    = self.FAST_directory
        fastBatch.FAST_InputFile    = self.FAST_InputFile  # FAST input file (ext=.fst)

        fastBatch.debug_level       = 2
        fastBatch.overwrite_outfiles = overwrite        # for debugging purposes
        
        # Set control parameters
        discon_vt = ROSCO_utilities.DISCON_dict(self.turbine,controller)
        for discon_input in discon_vt:
            self.case_inputs[('DISCON_in',discon_input)] = {'vals': [discon_vt[discon_input]], 'group': 0}
            
        self.case_list, self.case_name_list, self.dlc_list = self.iec.execute(case_inputs=self.case_inputs)
        
        fastBatch.case_list         = self.case_list
        fastBatch.case_name_list    = self.case_name_list
        fastBatch.channels          = self.channels
        fastBatch.FAST_runDirectory = self.FAST_level3_directory
        fastBatch.post              = FAST_IO_timeseries

        if self.n_cores == 1:
            out = fastBatch.run_serial()
        else:
            out = fastBatch.run_multi(cores=self.n_cores)

        self.level3_batch   = fastBatch
        self.level3_out     = out


class Level3_Turbine(object):
    
    def __init__(self,mf_turb):
        self.mf_turb = mf_turb

    def compute(self,omega_pc):
        self.mf_turb.controller.omega_pc = omega_pc
        self.mf_turb.run_level3(self.mf_turb.controller)

        return compute_outputs(self.mf_turb.level3_out)



class Level2_Turbine(object):

    def __init__(self,mf_turb,dofs):
        self.setup(mf_turb,dofs)

    def setup(self,mf_turb,dofs):
        
        # 1. Run linearization procedure, post-process linearization results, Load system from OpenFAST .lin files
        mf_turb.gen_level2_model(dofs)


        # 3. Run turbulent level 3 case, do this until we can extract rotor avg. wind speed directly from IEC cases
        # mf_turb.run_level3(mf_turb.turbine,mf_turb.controller)

        # 4. Extract disturbance from level3 sims (wind only for now)
        dist = []
        for case in mf_turb.case_list:
            ts_file     = TurbSimFile(case[('InflowWind','FileName_BTS')])
            ts_file.compute_rot_avg(mf_turb.iec.D/2)
            u_h         = ts_file['rot_avg'][0,:]
            tt          = ts_file['t']
            dist.append({'Time':tt, 'Wind': u_h})

        # save data to object
        self.disturbance    = dist
        self.mf_turb        = mf_turb

    def compute(self,omega_pc):
        # 5. Run level 2 simulation
        self.mf_turb.controller.omega_pc = omega_pc
        self.mf_turb.run_level2(self.mf_turb.controller,self.disturbance)

        outputs = compute_outputs(self.mf_turb.level2_out)

        return outputs

def compute_outputs(levelX_out):
    # compute Tower Base Myt DEL
    for lx_out in levelX_out:
        lx_out['meta'] = {}
        lx_out['meta']['name'] = 'placeholder'
    chan_info = [('TwrBsMyt',4)]
    la = Loads_Analysis()
    TwrBsMyt_DEL = la.get_DEL(levelX_out,chan_info)['TwrBsMyt'].tolist()

    # Generator Speed Measures
    GenSpeed_Max = [lx_out['GenSpeed'].max() for lx_out in levelX_out]
    GenSpeed_Std = [lx_out['GenSpeed'].std() for lx_out in levelX_out]

    # Platform pitch measures
    PtfmPitch_Max = [lx_out['PtfmPitch'].max() for lx_out in levelX_out]
    PtfmPitch_Std = [lx_out['PtfmPitch'].std() for lx_out in levelX_out]

    # save outputs
    outputs = {}
    outputs['TwrBsMyt_DEL']     = TwrBsMyt_DEL[0]
    outputs['GenSpeed_Max']     = GenSpeed_Max[0]
    outputs['GenSpeed_Std']     = GenSpeed_Std[0]
    outputs['PtfmPitch_Max']    = PtfmPitch_Max[0]
    outputs['PtfmPitch_Std']    = PtfmPitch_Std[0]

    return outputs

if __name__ == '__main__':
    # 0. Set up Model, using default input files
    import time
    s = time.time()

    mf_turb = MF_Turbine()
    mf_turb.n_cores = 4

    dofs = ['GenDOF','TwFADOF1','PtfmPDOF']
    # mf_turb.compare(dofs=dofs)

    l2_turb = Level2_Turbine(mf_turb, dofs)
    
    print('Time to train L2 model', time.time() - s)
    s = time.time()
    
    l2_outs = l2_turb.compute(.15)
    
    print('l2_outs')
    print(l2_outs)
    
    print()
    
    l2_outs = l2_turb.compute(.16)
    
    print('l2_outs changed')
    print(l2_outs)
    
    
    print('Time to compute two L2 models', time.time() - s)
    
    
    s = time.time()
    l3_turb = Level3_Turbine(mf_turb)
    
    print('Time to train L3 model', time.time() - s)
    s = time.time()
    
    l3_outs = l3_turb.compute(0.10082741)
    
    print('Time to compute L3 model', time.time() - s)
    s = time.time()
    
    print('l3_outs')
    print(l3_outs)
    
    # l3_turb = Level3_Turbine(mf_turb)
    # 
    # print()
    # 
    # l3_outs = l3_turb.compute(.16)
    # 
    # print('l3_outs changed')
    # print(l3_outs)
    # 
    # 
    
