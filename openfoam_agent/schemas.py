from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class GeometryType(str, Enum):
    BOX = "box"
    CYLINDER = "cylinder"
    PIPE = "pipe"
    AIRFOIL = "airfoil"
    BACKWARD_FACING_STEP = "backward_facing_step"
    LID_DRIVEN_CAVITY = "lid_driven_cavity"
    CHANNEL = "channel"
    SPHERE = "sphere"
    WEDGE = "wedge"
    PERIODIC_HILL = "periodic_hill"
    S_BEND = "s_bend"
    DIFFUSER = "diffuser"
    AHMED_BODY = "ahmed_body"
    MULTI_HILL = "multi_hill"
    T_JUNCTION = "t_junction"
    CD_NOZZLE = "cd_nozzle"
    ELBOW = "elbow"
    CUSTOM = "custom"


class TurbulenceModel(str, Enum):
    LAMINAR = "laminar"
    K_OMEGA_SST = "kOmegaSST"
    K_EPSILON = "kEpsilon"
    LES_SMAGORINSKY = "LesSmagorinsky"


class FlowRegime(str, Enum):
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


class CFDParams(BaseModel):
    geometry_type: GeometryType = Field(description="Primary geometry shape of the flow domain")
    is_3d: bool = Field(description="True if 3D simulation, False if 2D (thin extruded domain)")
    length: float = Field(default=1.0, description="Streamwise domain length in meters (must be > 0)")
    width: float = Field(default=1.0, description="Cross-stream width in meters (must be > 0)")
    height: float = Field(default=0.001, description="Domain height/depth in meters (use 0.001 for 2D empty-BC cases)")
    diameter: Optional[float] = Field(default=None, description="Characteristic diameter for cylinders/pipes in meters")
    reynolds_number: Optional[float] = Field(default=None, description="Reynolds number Re=U*L/nu. Null if not determinable.")
    inlet_velocity: float = Field(default=1.0, description="Inlet velocity magnitude in m/s (must be > 0)")
    outlet_pressure: float = Field(default=0.0, description="Outlet gauge pressure in Pa")
    kinematic_viscosity: float = Field(default=1.5e-5, description="Kinematic viscosity nu in m^2/s (1.5e-5 for air, 1e-6 for water)")
    density: float = Field(default=1.225, description="Fluid density in kg/m^3 (1.225 for air, 1000 for water)")
    flow_regime: FlowRegime = Field(description="Flow regime: laminar (Re<2300), transitional, or turbulent (Re>4000)")
    turbulence_model: TurbulenceModel = Field(description="Turbulence model to use")
    is_transient: bool = Field(description="True for time-dependent simulation, False for steady-state")
    is_compressible: bool = Field(default=False, description="True if density varies significantly")
    has_heat_transfer: bool = Field(default=False, description="True if temperature/buoyancy effects matter")
    is_multiphase: bool = Field(default=False, description="True for VOF/multiphase simulations")
    end_time: float = Field(description="Simulation end time in seconds (or pseudo-time steps for steady)")
    angle_of_attack: Optional[float] = Field(default=None, description="Angle of attack in degrees for airfoil cases")
    extraction_notes: str = Field(default="", description="Assumptions made during parameter extraction")


class RefinedPrompt(BaseModel):
    original: str
    refined: str
    added_context: str
    detected_ambiguities: list[str]


class RunResult(BaseModel):
    success: bool
    converged: bool
    runtime: float
    final_residuals: dict[str, float] = Field(default_factory=dict)
    residual_history: dict[str, list[float]] = Field(default_factory=dict)
    mesh_max_non_ortho: float = 0.0
    mesh_max_skewness: float = 0.0
    error_message: str = ""
    log: str = ""


class AgentResult(BaseModel):
    success: bool
    score: float
    feedback: str
    solver: str
    params: Optional[CFDParams] = None
    case_dir: str = ""
    error: str = ""
    runtime: float = 0.0
    attempt: int = 0
    residuals: dict[str, float] = Field(default_factory=dict)
    refined_prompt: str = ""
    rag_examples_used: list[str] = Field(default_factory=list)


class TrainingExample(BaseModel):
    prompt: str
    refined_prompt: str
    params: CFDParams
    case_dir: str
    solver: str
    score: float
    feedback: str
    converged: bool
    runtime: float
    timestamp: float
    case_files_text: str = ""
