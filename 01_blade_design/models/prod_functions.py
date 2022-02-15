import os
from collections import OrderedDict
import numpy as np
from weis.glue_code.runWEIS import run_weis
from wisdem.commonse.mpi_tools import MPI
from weis.multifidelity.models.base_model import BaseModel
from scipy.interpolate import PchipInterpolator
import dill
from wisdem.ccblade.ccblade import CCBlade as CCBladeOrig
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

## File management
run_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
fname_wt_input = run_dir + "IEA-15-240-RWT_WISDEMieaontology4all.yaml"
fname_modeling_options_ccblade = run_dir + "modeling_options_ccblade.yaml"
fname_modeling_options_openfast = run_dir + "modeling_options_openfast.yaml"
fname_modeling_options_olaf = run_dir + "modeling_options_olaf.yaml"
fname_modeling_options_openfast_AEP = run_dir + "modeling_options_openfast_AEP.yaml"
fname_analysis_options = run_dir + "analysis_options_prod.yaml"
folder_output = run_dir + "it_0/"
fname_wt_output = folder_output + "/temp.yaml"


class FullCCBlade(BaseModel):
    """
    Call the full WISDEM stack and focus on results from CCBlade.

    This calls WISDEM using the yaml files and all normal entry points, but
    has the additional overhead of unnecessary analyses.
    """

    def __init__(self, desvars_init, warmstart_file, turbine_geom_yaml=fname_wt_input, modeling_options=fname_modeling_options_ccblade):
        super(FullCCBlade, self).__init__(desvars_init, warmstart_file)
        self.turbine_geom_yaml = turbine_geom_yaml
        self.fname_modeling_options = modeling_options

    def compute(self, desvars):
        wt_opt_ccblade, analysis_options_ccblade, opt_options_ccblade = run_weis(
            self.turbine_geom_yaml,
            self.fname_modeling_options,
            fname_analysis_options,
            desvars,
        )
        
        outputs = {}
        outputs["AEP"] = wt_opt_ccblade.get_val("rp.AEP", units="GW*h")
        outputs["CP"] = wt_opt_ccblade["ccblade.CP"][0]
        outputs["power"] = wt_opt_ccblade["ccblade.P"][0]
        outputs["stall_check.no_stall_constraint"] = wt_opt_ccblade[
            "stall_check.no_stall_constraint"
        ]
        outputs["root_flapwise_bending_moment"] = wt_opt_ccblade.get_val(
            "ccblade.M", units="kN*m"
        )
        
        self.wt_opt = wt_opt_ccblade

        return outputs
        

class OpenFAST(BaseModel):
    """
    Call the full WISDEM stack and focus on results from OpenFAST.

    This calls WISDEM using the yaml files and all normal entry points, but
    has the additional overhead of unnecessary analyses. However, OpenFAST
    is generally more expensive than the other portions of WISDEM, making this
    less important.
    """

    def __init__(self, desvars_init, warmstart_file, turbine_geom_yaml=fname_wt_input,  modeling_options=fname_modeling_options_openfast):
        super(OpenFAST, self).__init__(desvars_init, warmstart_file)
        self.turbine_geom_yaml = turbine_geom_yaml
        self.fname_modeling_options = modeling_options

    def compute(self, desvars):
        wt_opt_openfast, analysis_options_openfast, opt_options_openfast = run_weis(
            self.turbine_geom_yaml,
            self.fname_modeling_options,
            fname_analysis_options,
            desvars,
        )

        outputs = {}
        outputs["CP"] = wt_opt_openfast["aeroelastic.Cp_out"][0]
        outputs["power"] = wt_opt_openfast["aeroelastic.P_aero_out"][0]
        outputs["stall_check.no_stall_constraint"] = wt_opt_openfast[
            "stall_check_of.no_stall_constraint"
        ]
        outputs["root_flapwise_bending_moment"] = wt_opt_openfast.get_val(
            "aeroelastic.max_RootMyb", units="kN*m"
        )[0]
        
        self.wt_opt = wt_opt_openfast

        return outputs
        
        

class OpenFASTAEP(BaseModel):
    """
    Call the full WISDEM stack and focus on results from OpenFAST.

    This calls WISDEM using the yaml files and all normal entry points, but
    has the additional overhead of unnecessary analyses. However, OpenFAST
    is generally more expensive than the other portions of WISDEM, making this
    less important.
    """

    def __init__(self, desvars_init, warmstart_file, turbine_geom_yaml=fname_wt_input,  modeling_options=fname_modeling_options_openfast_AEP):
        super(OpenFASTAEP, self).__init__(desvars_init, warmstart_file)
        self.turbine_geom_yaml = turbine_geom_yaml
        self.fname_modeling_options = modeling_options

    def compute(self, desvars):
        wt_opt_openfast, analysis_options_openfast, opt_options_openfast = run_weis(
            self.turbine_geom_yaml,
            self.fname_modeling_options,
            fname_analysis_options,
            desvars,
        )
        
        outputs = {}
        outputs["AEP"] = wt_opt_openfast.get_val("aeroelastic.AEP", units="GW*h")
        outputs["CP"] = wt_opt_openfast["aeroelastic.Cp_out"]
        outputs["power"] = wt_opt_openfast["aeroelastic.P_aero_out"]
        outputs["stall_check.no_stall_constraint"] = wt_opt_openfast[
            "stall_check_of.no_stall_constraint"
        ]
        outputs["root_flapwise_bending_moment"] = wt_opt_openfast.get_val(
            "aeroelastic.max_RootMyb", units="kN*m"
        )
        
        self.wt_opt = wt_opt_openfast
                
        return outputs


class OLAF(BaseModel):
    """
    Call the full WEIS stack and focus on results from OpenFAST using OLAF.

    This calls WEIS using the yaml files and all normal entry points, but
    has the additional overhead of unnecessary analyses. However, OpenFAST
    is generally more expensive than the other portions of WEIS, making this
    less important.
    """

    def __init__(self, desvars_init, warmstart_file, turbine_geom_yaml=fname_wt_input,  modeling_options=fname_modeling_options_olaf):
        super(OLAF, self).__init__(desvars_init, warmstart_file)
        self.turbine_geom_yaml = turbine_geom_yaml
        self.fname_modeling_options = modeling_options

    def compute(self, desvars):
        wt_opt_olaf, analysis_options_olaf, opt_options_olaf = run_weis(
            self.turbine_geom_yaml,
            self.fname_modeling_options,
            fname_analysis_options,
            desvars,
        )

        outputs = {}
        outputs["CP"] = wt_opt_olaf["aeroelastic.Cp_out"][0]
        outputs["power"] = wt_opt_olaf["aeroelastic.P_out"][0]
        outputs["stall_check.no_stall_constraint"] = wt_opt_olaf[
            "stall_check.no_stall_constraint"
        ]
        outputs["root_flapwise_bending_moment"] = wt_opt_olaf.get_val(
            "aeroelastic.max_RootMyb", units="kN*m"
        )[0]

        return outputs


class CCBlade(BaseModel):
    """
    Call only CCBlade as a standalone function using saved inputs.

    To ensure we're running the correct geometry, you need to first run
    FullCCBlade to save off some pickle files with info needed for this model.
    However, this model is much faster than the full WISDEM version because it
    doesn't call other analyses unnecessarily. For a quick test, this is
    about 320x faster.
    """

    def __init__(self, desvars_init, warmstart_file=None, n_span=30):
        super().__init__(desvars_init, warmstart_file)
        self.n_span = n_span

    def compute(self, desvars):
        with open(run_dir + f"CCBlade_inputs_{self.n_span}.pkl", "rb") as f:
            saved_dict = dill.load(f)

        chord_opt_gain = desvars["blade.opt_var.chord_opt_gain"]

        chord_original = saved_dict["chord_original"]
        s = saved_dict["s"]
        s_opt_chord = np.linspace(0.0, 1.0, len(chord_opt_gain))

        spline = PchipInterpolator
        chord_spline = spline(s_opt_chord, chord_opt_gain)
        chord = chord_original * chord_spline(s)

        get_cp_cm = CCBladeOrig(
            saved_dict["r"],
            chord,
            saved_dict["twist"],
            saved_dict["af"],
            saved_dict["Rhub"],
            saved_dict["Rtip"],
            saved_dict["nBlades"],
            saved_dict["rho"],
            saved_dict["mu"],
            saved_dict["precone"],
            saved_dict["tilt"],
            saved_dict["yaw"],
            saved_dict["shearExp"],
            saved_dict["hub_height"],
            saved_dict["nSector"],
            saved_dict["precurve"],
            saved_dict["precurveTip"],
            saved_dict["presweep"],
            saved_dict["presweepTip"],
            saved_dict["tiploss"],
            saved_dict["hubloss"],
            saved_dict["wakerotation"],
            saved_dict["usecd"],
        )
        get_cp_cm.inverse_analysis = False
        get_cp_cm.induction = True

        # Compute omega given TSR
        Omega = (
            saved_dict["Uhub"] * saved_dict["tsr"] / saved_dict["Rtip"] * 30.0 / np.pi
        )

        myout, derivs = get_cp_cm.evaluate(
            [saved_dict["Uhub"]], [Omega], [saved_dict["pitch"]], coefficients=True
        )

        outputs = {}
        outputs["CP"] = myout["CP"]

        return outputs
