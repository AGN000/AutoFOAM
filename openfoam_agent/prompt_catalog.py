"""
Augmented prompt catalog for OpenFOAM expert fine-tuning data generation.

Each entry pairs a natural-language prompt with ground-truth CFDParams so the
simulation pipeline can be run without an LLM for param extraction.

Solver coverage (via solver_selector.py):
  simpleFoam          — steady incompressible (default)
  icoFoam             — transient laminar (is_transient=True, Re<2300, LAMINAR)
  pimpleFoam          — transient turbulent/high-Re (is_transient=True, else)
  buoyantSimpleFoam   — steady with heat transfer (has_heat_transfer=True)
  buoyantPimpleFoam   — transient heat transfer (has_heat_transfer=True, is_transient=True)
  rhoSimpleFoam       — steady compressible (is_compressible=True)
  rhoPimpleFoam       — transient compressible (is_compressible=True, is_transient=True)
  interFoam           — VOF multiphase (is_multiphase=True)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import (
    CFDParams, GeometryType, FlowRegime, TurbulenceModel,
)


@dataclass
class PromptCase:
    """One (prompt, params) training pair."""
    prompt: str
    params: CFDParams
    case_tag: str          # e.g. "cavity_re100"
    expert_notes: str = "" # physics notes for expert response


def _cavity(re: float, size: float = 1.0, fluid: str = "air") -> CFDParams:
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    u_lid = re * nu / size
    return CFDParams(
        geometry_type=GeometryType.LID_DRIVEN_CAVITY,
        is_3d=False,
        length=size, width=size, height=0.001,
        diameter=None,
        reynolds_number=re,
        inlet_velocity=u_lid,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.LAMINAR if re < 2300 else FlowRegime.TURBULENT,
        turbulence_model=TurbulenceModel.LAMINAR if re < 2300 else TurbulenceModel.K_OMEGA_SST,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=2000.0,
        extraction_notes=f"lid-driven cavity Re={re}",
    )


def _pipe(re: float, D: float, L: float, fluid: str = "air", is_3d: bool = True) -> CFDParams:
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    u_inlet = re * nu / D
    turb = re > 4000
    return CFDParams(
        geometry_type=GeometryType.PIPE,
        is_3d=is_3d,
        length=L, width=D, height=D,
        diameter=D,
        reynolds_number=re,
        inlet_velocity=u_inlet,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=1000.0,
        extraction_notes=f"pipe flow Re={re} D={D}m L={L}m",
    )


def _cylinder(re: float, D: float, fluid: str = "air") -> CFDParams:
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    u_inlet = re * nu / D
    domain_L = max(20 * D, 2.0)
    domain_W = max(8 * D, 0.8)
    return CFDParams(
        geometry_type=GeometryType.CYLINDER,
        is_3d=False,
        length=domain_L, width=domain_W, height=0.001,
        diameter=D,
        reynolds_number=re,
        inlet_velocity=u_inlet,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.LAMINAR if re < 2300 else FlowRegime.TURBULENT,
        turbulence_model=TurbulenceModel.LAMINAR if re < 2300 else TurbulenceModel.K_OMEGA_SST,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=1000.0,
        extraction_notes=f"cylinder flow Re={re} D={D}m",
    )


def _channel(re: float, L: float, H: float, fluid: str = "air") -> CFDParams:
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    # Re for channel based on half-height (H/2)
    u_inlet = re * nu / (H / 2)
    turb = re > 4000
    return CFDParams(
        geometry_type=GeometryType.CHANNEL,
        is_3d=False,
        length=L, width=H, height=0.001,
        diameter=None,
        reynolds_number=re,
        inlet_velocity=u_inlet,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=1000.0,
        extraction_notes=f"channel flow Re={re} L={L}m H={H}m",
    )


def _bfs(re: float, step_h: float, L_up: float, L_down: float) -> CFDParams:
    nu = 1.5e-5
    rho = 1.225
    u_inlet = re * nu / step_h
    turb = re > 4000
    return CFDParams(
        geometry_type=GeometryType.BACKWARD_FACING_STEP,
        is_3d=False,
        length=L_up + L_down, width=2 * step_h, height=0.001,
        diameter=None,
        reynolds_number=re,
        inlet_velocity=u_inlet,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=1000.0,
        extraction_notes=f"backward-facing step Re={re} h={step_h}m",
    )


def _airfoil(re: float, chord: float, aoa: float, fluid: str = "air") -> CFDParams:
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    U = re * nu / chord
    turb = re > 5000
    return CFDParams(
        geometry_type=GeometryType.AIRFOIL,
        is_3d=False,
        length=chord, width=20 * chord, height=0.001,
        diameter=None,
        angle_of_attack=aoa,
        reynolds_number=re,
        inlet_velocity=U,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=2000.0,
        extraction_notes=f"NACA0012 airfoil Re={re:.0e} AoA={aoa}deg chord={chord}m",
    )


def _wedge(re: float, D: float, L: float, fluid: str = "air") -> CFDParams:
    """Axisymmetric pipe using 5-degree wedge geometry."""
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    U = re * nu / D
    turb = re > 4000
    return CFDParams(
        geometry_type=GeometryType.WEDGE,
        is_3d=True,
        length=L, width=D, height=D,
        diameter=D,
        reynolds_number=re,
        inlet_velocity=U,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=1000.0,
        extraction_notes=f"axisymmetric wedge pipe Re={re} D={D}m L={L}m",
    )


def _box(re: float, L: float, H: float, is_3d: bool = False, fluid: str = "air") -> CFDParams:
    """General rectangular box / flat-plate channel flow."""
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    U = re * nu / H
    turb = re > 4000
    return CFDParams(
        geometry_type=GeometryType.BOX,
        is_3d=is_3d,
        length=L, width=H, height=0.001 if not is_3d else H,
        diameter=None,
        reynolds_number=re,
        inlet_velocity=U,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=1000.0,
        extraction_notes=f"box/duct flow Re={re} L={L}m H={H}m",
    )


def _cavity_transient(re: float, size: float = 1.0) -> CFDParams:
    """Transient lid-driven cavity → icoFoam (laminar, Re<2300)."""
    nu = 1.5e-5
    u_lid = re * nu / size
    return CFDParams(
        geometry_type=GeometryType.LID_DRIVEN_CAVITY,
        is_3d=False,
        length=size, width=size, height=0.001,
        diameter=None,
        reynolds_number=re,
        inlet_velocity=u_lid,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=1.225,
        flow_regime=FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_transient=True,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=10.0,
        extraction_notes=f"transient lid-driven cavity Re={re} → icoFoam",
    )


def _cylinder_transient(re: float, D: float, fluid: str = "air") -> CFDParams:
    """Transient cylinder flow: icoFoam if laminar Re<2300, pimpleFoam otherwise."""
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    U = re * nu / D
    domain_L = max(20 * D, 2.0)
    domain_W = max(8 * D, 0.8)
    turb = re > 2300
    return CFDParams(
        geometry_type=GeometryType.CYLINDER,
        is_3d=False,
        length=domain_L, width=domain_W, height=0.001,
        diameter=D,
        reynolds_number=re,
        inlet_velocity=U,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=True,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=20.0,
        extraction_notes=f"transient cylinder Re={re} D={D}m",
    )


def _pipe_transient(re: float, D: float, L: float, fluid: str = "air") -> CFDParams:
    """Transient pipe startup: icoFoam (laminar) or pimpleFoam (turbulent)."""
    nu = 1.5e-5 if fluid == "air" else 1e-6
    rho = 1.225 if fluid == "air" else 1000.0
    U = re * nu / D
    turb = re > 2300
    return CFDParams(
        geometry_type=GeometryType.PIPE,
        is_3d=True,
        length=L, width=D, height=D,
        diameter=D,
        reynolds_number=re,
        inlet_velocity=U,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=rho,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=True,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=5.0,
        extraction_notes=f"transient pipe Re={re} D={D}m L={L}m",
    )


def _compressible_box(mach: float, L: float, H: float, is_transient: bool = True) -> CFDParams:
    """High-speed channel/box → rhoPimpleFoam (transient) or rhoSimpleFoam (steady)."""
    gamma, R = 1.4, 287.0
    T_ref = 300.0
    c_sound = (gamma * R * T_ref) ** 0.5  # ~347 m/s
    U = mach * c_sound
    nu = 1.5e-5
    Re = U * H / nu
    turb = Re > 4000
    return CFDParams(
        geometry_type=GeometryType.BOX,
        is_3d=False,
        length=L, width=H, height=0.001,
        diameter=None,
        reynolds_number=Re,
        inlet_velocity=U,
        outlet_pressure=101325.0,
        kinematic_viscosity=nu,
        density=1.225,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=is_transient,
        is_compressible=True,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=0.05 if is_transient else 500.0,
        extraction_notes=f"compressible box Ma={mach:.1f} L={L}m H={H}m",
    )


def _compressible_pipe(mach: float, D: float, L: float, is_transient: bool = False) -> CFDParams:
    """Compressible pipe flow → rhoSimpleFoam (steady) or rhoPimpleFoam (transient)."""
    gamma, R = 1.4, 287.0
    T_ref = 300.0
    c_sound = (gamma * R * T_ref) ** 0.5
    U = mach * c_sound
    nu = 1.5e-5
    Re = U * D / nu
    turb = Re > 4000
    return CFDParams(
        geometry_type=GeometryType.PIPE,
        is_3d=True,
        length=L, width=D, height=D,
        diameter=D,
        reynolds_number=Re,
        inlet_velocity=U,
        outlet_pressure=101325.0,
        kinematic_viscosity=nu,
        density=1.225,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=is_transient,
        is_compressible=True,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=0.05 if is_transient else 500.0,
        extraction_notes=f"compressible pipe Ma={mach:.1f} D={D}m L={L}m",
    )


def _dam_break(L: float = 4.0, H: float = 2.0) -> CFDParams:
    """Water dam break in air → interFoam (VOF multiphase transient)."""
    return CFDParams(
        geometry_type=GeometryType.BOX,
        is_3d=False,
        length=L, width=H, height=0.001,
        diameter=None,
        reynolds_number=50000.0,
        inlet_velocity=0.01,
        outlet_pressure=0.0,
        kinematic_viscosity=1e-6,
        density=1000.0,
        flow_regime=FlowRegime.TURBULENT,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_transient=True,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=True,
        end_time=1.0,
        extraction_notes=f"dam break VOF L={L}m H={H}m → interFoam",
    )


def _wave_channel(L: float = 10.0, H: float = 2.0) -> CFDParams:
    """Free-surface wave channel → interFoam (VOF multiphase transient)."""
    return CFDParams(
        geometry_type=GeometryType.CHANNEL,
        is_3d=False,
        length=L, width=H, height=0.001,
        diameter=None,
        reynolds_number=10000.0,
        inlet_velocity=1.0,
        outlet_pressure=0.0,
        kinematic_viscosity=1e-6,
        density=1000.0,
        flow_regime=FlowRegime.TURBULENT,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_transient=True,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=True,
        end_time=5.0,
        extraction_notes=f"wave channel VOF L={L}m H={H}m → interFoam",
    )


def _cavity_buoyancy(size: float = 1.0) -> CFDParams:
    """Differentially heated cavity → buoyantSimpleFoam."""
    return CFDParams(
        geometry_type=GeometryType.LID_DRIVEN_CAVITY,
        is_3d=False,
        length=size, width=size, height=0.001,
        diameter=None,
        reynolds_number=100.0,
        inlet_velocity=0.01,
        outlet_pressure=0.0,
        kinematic_viscosity=1.5e-5,
        density=1.225,
        flow_regime=FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=True,
        is_multiphase=False,
        end_time=2000.0,
        extraction_notes="buoyancy-driven cavity → buoyantSimpleFoam",
    )


def _buoyancy(size: float = 1.0, turb: bool = False, end_time: float = 2000.0) -> CFDParams:
    """Differentially heated cavity → buoyantSimpleFoam (laminar or turbulent Ra)."""
    return CFDParams(
        geometry_type=GeometryType.LID_DRIVEN_CAVITY,
        is_3d=False,
        length=size, width=size, height=0.001,
        diameter=None,
        reynolds_number=1000.0 if turb else 100.0,
        inlet_velocity=0.01,
        outlet_pressure=0.0,
        kinematic_viscosity=1.5e-5,
        density=1.225,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=False,
        is_compressible=False,
        has_heat_transfer=True,
        is_multiphase=False,
        end_time=end_time,
        extraction_notes=f"buoyancy cavity size={size}m {'turb' if turb else 'lam'} → buoyantSimpleFoam",
    )


def _channel_transient(re: float, L: float, H: float) -> CFDParams:
    """Transient channel flow → icoFoam (Re<2300) or pimpleFoam (Re>=2300)."""
    nu = 1.5e-5
    u_inlet = re * nu / (H / 2)
    turb = re >= 2300
    return CFDParams(
        geometry_type=GeometryType.CHANNEL,
        is_3d=False,
        length=L, width=H, height=0.001,
        diameter=None,
        reynolds_number=re,
        inlet_velocity=u_inlet,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=1.225,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=True,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=10.0,
        extraction_notes=f"transient channel Re={re} L={L}m H={H}m",
    )


def _bfs_transient(re: float, step_h: float) -> CFDParams:
    """Transient backward-facing step → icoFoam (Re<2300) or pimpleFoam (Re>=2300)."""
    nu = 1.5e-5
    u_inlet = re * nu / step_h
    turb = re >= 2300
    return CFDParams(
        geometry_type=GeometryType.BACKWARD_FACING_STEP,
        is_3d=False,
        length=0.2 + 2.0 * step_h, width=2 * step_h, height=0.001,
        diameter=None,
        reynolds_number=re,
        inlet_velocity=u_inlet,
        outlet_pressure=0.0,
        kinematic_viscosity=nu,
        density=1.225,
        flow_regime=FlowRegime.TURBULENT if turb else FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.K_OMEGA_SST if turb else TurbulenceModel.LAMINAR,
        is_transient=True,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=False,
        end_time=10.0,
        extraction_notes=f"transient BFS Re={re} h={step_h}m",
    )


def _sloshing(L: float = 2.0, H: float = 1.0) -> CFDParams:
    """Partially-filled sloshing tank → interFoam (VOF transient)."""
    return CFDParams(
        geometry_type=GeometryType.BOX,
        is_3d=False,
        length=L, width=H, height=0.001,
        diameter=None,
        reynolds_number=10000.0,
        inlet_velocity=0.01,
        outlet_pressure=0.0,
        kinematic_viscosity=1e-6,
        density=1000.0,
        flow_regime=FlowRegime.TURBULENT,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_transient=True,
        is_compressible=False,
        has_heat_transfer=False,
        is_multiphase=True,
        end_time=2.0,
        extraction_notes=f"sloshing tank VOF L={L}m H={H}m → interFoam",
    )


# ---------------------------------------------------------------------------
# Augmented prompt catalog
# ---------------------------------------------------------------------------

PROMPT_CATALOG: list[PromptCase] = [

    # ── LID-DRIVEN CAVITY ──────────────────────────────────────────────────

    PromptCase(
        prompt="2D lid-driven cavity flow at Re=100",
        params=_cavity(100),
        case_tag="cavity_re100_a",
        expert_notes="Classic Ghia benchmark. Laminar, steady. Single-cell depth with empty BCs.",
    ),
    PromptCase(
        prompt="simulate a square cavity with a moving top wall at Reynolds number 100, fluid is air",
        params=_cavity(100),
        case_tag="cavity_re100_b",
        expert_notes="Lid-driven cavity Re=100. simpleFoam steady laminar.",
    ),
    PromptCase(
        prompt="lid cavity benchmark case, Re=100, 1m × 1m domain, steady state",
        params=_cavity(100),
        case_tag="cavity_re100_c",
        expert_notes="Lid-driven cavity Re=100 Ghia benchmark.",
    ),
    PromptCase(
        prompt="OpenFOAM cavity tutorial: 2D square driven cavity, Re=100, kinematic viscosity 0.01 m2/s",
        params=_cavity(100, fluid="air"),
        case_tag="cavity_re100_d",
        expert_notes="Classic isobaric driven cavity. Re=100, laminar.",
    ),
    PromptCase(
        prompt="lid-driven cavity flow at Re=400 in a 1m square domain",
        params=_cavity(400),
        case_tag="cavity_re400",
        expert_notes="Re=400 still laminar but stronger recirculation corners visible.",
    ),
    PromptCase(
        prompt="square cavity with lid velocity, Re=1000, air, steady simpleFoam",
        params=_cavity(1000),
        case_tag="cavity_re1000",
        expert_notes="Re=1000 cavity. Laminar but complex vortex structure.",
    ),
    PromptCase(
        prompt="driven cavity simulation at Re=3200, 2D, steady-state with air",
        params=_cavity(3200),
        case_tag="cavity_re3200",
        expert_notes="Re=3200 approaching transitional. kOmegaSST chosen conservatively.",
    ),
    PromptCase(
        prompt="benchmark cavity flow, lid velocity U=1 m/s, 1m square, nu=0.001 m2/s (Re=1000)",
        params=_cavity(1000),
        case_tag="cavity_re1000_explicit",
        expert_notes="Re=1000, explicitly given velocity and viscosity.",
    ),
    PromptCase(
        prompt="2D lid-driven cavity, Re=100, water as fluid (nu=1e-6 m2/s, rho=1000 kg/m3)",
        params=_cavity(100, fluid="water"),
        case_tag="cavity_re100_water",
        expert_notes="Same Re but water properties. Very low lid velocity U=1e-4 m/s.",
    ),
    PromptCase(
        prompt="cavity flow validation case: 1×1 m square, moving top lid, Reynolds=100, 2D incompressible",
        params=_cavity(100),
        case_tag="cavity_re100_e",
        expert_notes="Validation case. Compare against Ghia 1982 data.",
    ),

    # ── PIPE FLOW ─────────────────────────────────────────────────────────

    PromptCase(
        prompt="turbulent pipe flow Re=50000, diameter=0.05m, length=0.5m",
        params=_pipe(50000, 0.05, 0.5),
        case_tag="pipe_re50k",
        expert_notes="Fully turbulent, kOmegaSST with wall functions. 3D circular cross-section.",
    ),
    PromptCase(
        prompt="3D pipe flow simulation: diameter 5 cm, length 50 cm, Re=50000, air, turbulent",
        params=_pipe(50000, 0.05, 0.5),
        case_tag="pipe_re50k_b",
        expert_notes="Same as canonical pipe but specified in cm.",
    ),
    PromptCase(
        prompt="fully developed turbulent flow in a circular pipe, D=5cm, L=0.5m, Re=100000",
        params=_pipe(100000, 0.05, 0.5),
        case_tag="pipe_re100k",
        expert_notes="High Re turbulent pipe. kOmegaSST, wall functions, y+ target 30.",
    ),
    PromptCase(
        prompt="laminar pipe flow, Re=500, tube diameter 2cm, length 30cm",
        params=_pipe(500, 0.02, 0.3),
        case_tag="pipe_re500_lam",
        expert_notes="Laminar Hagen-Poiseuille flow. No turbulence model needed.",
    ),
    PromptCase(
        prompt="Hagen-Poiseuille flow verification: circular pipe, D=2cm, L=0.3m, Re=500, air",
        params=_pipe(500, 0.02, 0.3),
        case_tag="pipe_re500_hp",
        expert_notes="Laminar pipe flow. Validate parabolic velocity profile.",
    ),
    PromptCase(
        prompt="pipe flow Re=1000, D=0.02m, L=0.3m, steady, incompressible air",
        params=_pipe(1000, 0.02, 0.3),
        case_tag="pipe_re1000",
        expert_notes="Laminar pipe flow. Entrance length L_entry ≈ 0.06·Re·D = 1.2m, so flow not fully developed.",
    ),
    PromptCase(
        prompt="turbulent water pipe flow, Re=20000, diameter=50mm, length=500mm",
        params=_pipe(20000, 0.05, 0.5, fluid="water"),
        case_tag="pipe_re20k_water",
        expert_notes="Turbulent water pipe. nu=1e-6, rho=1000. kOmegaSST.",
    ),
    PromptCase(
        prompt="pipe flow at Re=10000, D=0.05m, L=0.5m, turbulent, kOmegaSST, air",
        params=_pipe(10000, 0.05, 0.5),
        case_tag="pipe_re10k",
        expert_notes="Turbulent pipe flow, moderate Re.",
    ),
    PromptCase(
        prompt="internal flow in a round duct: diameter 50mm, length 500mm, inlet velocity 15 m/s, air",
        params=_pipe(50000, 0.05, 0.5),  # Re = 15*0.05/1.5e-5 = 50000
        case_tag="pipe_u15_air",
        expert_notes="Re=50000 from U=15 m/s, D=50mm, nu=1.5e-5.",
    ),
    PromptCase(
        prompt="develop turbulent boundary layer in a pipe: Re=50000, D=5cm, L=0.5m, 3D, kOmegaSST, wall functions",
        params=_pipe(50000, 0.05, 0.5),
        case_tag="pipe_re50k_c",
        expert_notes="Expert phrasing emphasising wall treatment.",
    ),

    # ── CYLINDER FLOW ─────────────────────────────────────────────────────

    PromptCase(
        prompt="2D flow over a circular cylinder at Re=200, D=0.1m",
        params=_cylinder(200, 0.1),
        case_tag="cyl_re200",
        expert_notes="Re=200 laminar steady wake. No vortex shedding at steady-state.",
    ),
    PromptCase(
        prompt="flow around a cylinder, Re=200, diameter 10cm, 2D, steady, air",
        params=_cylinder(200, 0.1),
        case_tag="cyl_re200_b",
        expert_notes="Bluff body benchmark. Domain: 20D long, 8D wide.",
    ),
    PromptCase(
        prompt="circular cylinder in cross-flow, D=0.1m, Re=100, incompressible 2D",
        params=_cylinder(100, 0.1),
        case_tag="cyl_re100",
        expert_notes="Re=100 attached laminar flow, no separation.",
    ),
    PromptCase(
        prompt="cylinder wake simulation at Re=500, diameter=0.05m, 2D steady",
        params=_cylinder(500, 0.05),
        case_tag="cyl_re500",
        expert_notes="Re=500 steady laminar. Vortex shedding would need transient.",
    ),
    PromptCase(
        prompt="von Karman vortex street: cylinder Re=200, D=0.1m, 2D, steady simpleFoam",
        params=_cylinder(200, 0.1),
        case_tag="cyl_re200_karman",
        expert_notes="For vortex street need pimpleFoam transient; simpleFoam captures mean drag.",
    ),
    PromptCase(
        prompt="drag coefficient estimation for a 2D cylinder, Re=200, D=0.1m, air",
        params=_cylinder(200, 0.1),
        case_tag="cyl_re200_drag",
        expert_notes="Post-process Cd from pressure and viscous forces on cylinder patch.",
    ),
    PromptCase(
        prompt="flow past circular cylinder, Re=40, D=10cm, 2D laminar steady",
        params=_cylinder(40, 0.1),
        case_tag="cyl_re40",
        expert_notes="Re=40 fully attached laminar flow, Cd ≈ 1.5.",
    ),
    PromptCase(
        prompt="cylinder in external flow: diameter=0.2m, Re=200, 2D air, domain 4m long × 1.6m wide",
        params=_cylinder(200, 0.2),
        case_tag="cyl_re200_d02",
        expert_notes="Same Re, larger cylinder. Domain scales with D.",
    ),
    PromptCase(
        prompt="2D cylinder cross-flow Re=200, air, nu=1.5e-5 m2/s, D=0.1m, steady incompressible",
        params=_cylinder(200, 0.1),
        case_tag="cyl_re200_c",
        expert_notes="Explicit nu given.",
    ),
    PromptCase(
        prompt="OpenFOAM simpleFoam: external flow over cylinder, Re=200, diameter 100mm, 2D",
        params=_cylinder(200, 0.1),
        case_tag="cyl_re200_e",
        expert_notes="Solver explicitly named by user.",
    ),

    # ── CHANNEL FLOW ──────────────────────────────────────────────────────

    PromptCase(
        prompt="2D turbulent channel flow at Re=10000, length 5m, height 0.1m",
        params=_channel(10000, 5.0, 0.1),
        case_tag="chan_re10k",
        expert_notes="Turbulent channel, Re_tau ≈ 360. kOmegaSST with wall functions.",
    ),
    PromptCase(
        prompt="plane channel flow simulation, Re=5000, L=5m, H=0.1m, 2D, air",
        params=_channel(5000, 5.0, 0.1),
        case_tag="chan_re5k",
        expert_notes="Turbulent channel. Fully developed inflow assumption.",
    ),
    PromptCase(
        prompt="fully developed turbulent channel flow: Re=20000, height=0.2m, length=4m, 2D",
        params=_channel(20000, 4.0, 0.2),
        case_tag="chan_re20k",
        expert_notes="High Re turbulent channel. Log-law wall region.",
    ),
    PromptCase(
        prompt="laminar channel flow Re=1000, L=10m, H=0.1m, verify Poiseuille profile",
        params=_channel(1000, 10.0, 0.1),
        case_tag="chan_re1k_lam",
        expert_notes="Laminar Poiseuille. Parabolic profile u_max = 1.5 * U_mean.",
    ),
    PromptCase(
        prompt="2D incompressible channel flow, air, Re=2000, length 8m, channel height 0.08m",
        params=_channel(2000, 8.0, 0.08),
        case_tag="chan_re2k",
        expert_notes="Transitional Re. Using laminar solver for conservative approach.",
    ),
    PromptCase(
        prompt="rectangular duct flow 2D, Re=50000, H=0.1m, L=5m, turbulent kOmegaSST",
        params=_channel(50000, 5.0, 0.1),
        case_tag="chan_re50k",
        expert_notes="High Re channel. Strong wall shear, y+ must target 30 for wall functions.",
    ),
    PromptCase(
        prompt="turbulent channel simulation: Re=10000, periodic in streamwise, height=0.1m",
        params=_channel(10000, 5.0, 0.1),
        case_tag="chan_re10k_periodic",
        expert_notes="Periodic BCs would be ideal; simpleFoam with inlet/outlet approximation.",
    ),
    PromptCase(
        prompt="plane Couette-Poiseuille flow 2D channel Re=3000, L=5m, H=0.1m",
        params=_channel(3000, 5.0, 0.1),
        case_tag="chan_re3k",
        expert_notes="Mixed-driven flow, Re approaching turbulent transition.",
    ),
    PromptCase(
        prompt="2D air channel, centerline velocity 1.5 m/s, H=0.1m, L=5m, nu=1.5e-5 m2/s",
        params=_channel(10000, 5.0, 0.1),  # Re ~ U_mean*H/2/nu = 1.0*0.05/1.5e-5 ≈ 3333
        case_tag="chan_u15",
        expert_notes="Re specified via velocity. U_inlet = U_mean ≈ 1 m/s for parabolic profile.",
    ),
    PromptCase(
        prompt="steady 2D channel flow, Re=10000, develop turbulent boundary layers on top and bottom walls",
        params=_channel(10000, 5.0, 0.1),
        case_tag="chan_re10k_b",
        expert_notes="kOmegaSST with kqRWallFunction on top/bottom walls.",
    ),

    # ── BACKWARD-FACING STEP ──────────────────────────────────────────────

    PromptCase(
        prompt="backward-facing step flow at Re=800, step height h=0.1m",
        params=_bfs(800, 0.1, 0.2, 2.0),
        case_tag="bfs_re800",
        expert_notes="Classic Armaly benchmark Re=800. Laminar, reattachment length ≈ 6h.",
    ),
    PromptCase(
        prompt="flow over a backward facing step: Re=800, h=0.1m, 2D, steady, air",
        params=_bfs(800, 0.1, 0.2, 2.0),
        case_tag="bfs_re800_b",
        expert_notes="Backward-facing step Re=800 benchmark. Recirculation region downstream.",
    ),
    PromptCase(
        prompt="backward step flow, Re=200, step height=0.05m, upstream length=0.2m, downstream=1m",
        params=_bfs(200, 0.05, 0.2, 1.0),
        case_tag="bfs_re200",
        expert_notes="Low Re BFS. Smaller recirculation, reattachment ≈ 3h.",
    ),
    PromptCase(
        prompt="turbulent backward-facing step Re=5000, h=0.1m, 2D, kOmegaSST",
        params=_bfs(5000, 0.1, 0.2, 3.0),
        case_tag="bfs_re5k",
        expert_notes="Turbulent BFS. kOmegaSST, wall functions on step and bottom wall.",
    ),
    PromptCase(
        prompt="BFS benchmark case: step height 10cm, Re=800, 2D incompressible air, simpleFoam",
        params=_bfs(800, 0.1, 0.2, 2.0),
        case_tag="bfs_re800_c",
        expert_notes="Standard BFS. Expansion ratio 2:1.",
    ),
    PromptCase(
        prompt="flow separation at a backward step, Re=800, h=0.1m, 2D, compute reattachment length",
        params=_bfs(800, 0.1, 0.2, 2.0),
        case_tag="bfs_re800_sep",
        expert_notes="Reattachment length measured from step face to zero wall-shear line.",
    ),
    PromptCase(
        prompt="backward-facing step 2D Re=1000, h=0.1m, steady incompressible, air",
        params=_bfs(1000, 0.1, 0.2, 2.0),
        case_tag="bfs_re1000",
        expert_notes="Slightly higher Re than Armaly. Still laminar.",
    ),
    PromptCase(
        prompt="step flow: upstream height h=0.1m, downstream 2h, Re=800, steady 2D air",
        params=_bfs(800, 0.1, 0.2, 2.0),
        case_tag="bfs_re800_d",
        expert_notes="Expansion ratio 1:2. Standard setup.",
    ),

    # ── AIRFOIL — simpleFoam (turbulent/laminar) ───────────────────────────

    PromptCase(
        prompt="RANS simulation of NACA0012 airfoil, Re=1e6, angle of attack 5 degrees, chord=1m, kOmegaSST",
        params=_airfoil(1e6, 1.0, 5.0),
        case_tag="airfoil_re1m_aoa5",
        expert_notes="Canonical turbulent airfoil case. Far-field box 20c. simpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="simulate airfoil flow at Reynolds number 1 million, angle of attack 5 degrees, air",
        params=_airfoil(1e6, 1.0, 5.0),
        case_tag="airfoil_re1m_aoa5_b",
        expert_notes="Same as canonical but beginner phrasing — agent must infer kOmegaSST.",
    ),
    PromptCase(
        prompt="NACA0012 airfoil, chord=0.5m, Re=500000, AoA=3 degrees, steady incompressible air",
        params=_airfoil(5e5, 0.5, 3.0),
        case_tag="airfoil_re500k_aoa3",
        expert_notes="Mid Re turbulent airfoil. U=15 m/s, chord=0.5m.",
    ),
    PromptCase(
        prompt="low-Reynolds airfoil: NACA0012, chord=0.1m, Re=10000, AoA=2 degrees, laminar",
        params=_airfoil(1e4, 0.1, 2.0),
        case_tag="airfoil_re1e4_lam",
        expert_notes="Laminar airfoil flow Re=10000. No turbulence model needed.",
    ),
    PromptCase(
        prompt="high-Re airfoil NACA0012: Re=2 million, chord 1m, angle of attack 10 degrees, turbulent",
        params=_airfoil(2e6, 1.0, 10.0),
        case_tag="airfoil_re2m_aoa10",
        expert_notes="High Re with significant adverse pressure gradient near trailing edge.",
    ),
    PromptCase(
        prompt="NACA0012 at zero angle of attack, Re=1e6, chord=1m, measure lift and drag coefficients",
        params=_airfoil(1e6, 1.0, 0.0),
        case_tag="airfoil_re1m_aoa0",
        expert_notes="Zero incidence — symmetric flow, Cl=0 expected. Cd validation.",
    ),
    PromptCase(
        prompt="airfoil simulation: chord 1m, Re=300000, AoA=8 degrees, air, kOmegaSST RANS",
        params=_airfoil(3e5, 1.0, 8.0),
        case_tag="airfoil_re300k_aoa8",
        expert_notes="Moderate turbulent airfoil, angle approaching maximum lift.",
    ),
    PromptCase(
        prompt="NACA0012 near stall: Re=1e6, chord=1m, angle of attack=15 degrees, turbulent 2D",
        params=_airfoil(1e6, 1.0, 15.0),
        case_tag="airfoil_re1m_aoa15",
        expert_notes="High AoA near stall. Flow separation on suction side expected.",
    ),
    PromptCase(
        prompt="2D airfoil CFD: NACA0012, Re=50000, AoA=5 degrees, chord=0.1m",
        params=_airfoil(5e4, 0.1, 5.0),
        case_tag="airfoil_re5e4_aoa5",
        expert_notes="Transitional Re. kOmegaSST still chosen (Re>5000 threshold).",
    ),
    PromptCase(
        prompt="OpenFOAM simpleFoam RANS: NACA0012, chord 1m, freestream 15 m/s, AoA=5deg, air nu=1.5e-5",
        params=_airfoil(1e6, 1.0, 5.0),
        case_tag="airfoil_re1m_explicit",
        expert_notes="All parameters explicit — Re=1e6 from U=15, chord=1m, nu=1.5e-5.",
    ),

    # ── WEDGE (axisymmetric) — simpleFoam ─────────────────────────────────

    PromptCase(
        prompt="axisymmetric pipe flow using wedge geometry, Re=500, D=0.02m, L=0.3m, laminar air",
        params=_wedge(500, 0.02, 0.3),
        case_tag="wedge_re500_lam",
        expert_notes="Axisymmetric Hagen-Poiseuille. Wedge 5deg, axis BC on centreline.",
    ),
    PromptCase(
        prompt="OpenFOAM wedge simulation: axisymmetric laminar pipe, diameter=20mm, length=300mm, Re=1000",
        params=_wedge(1000, 0.02, 0.3),
        case_tag="wedge_re1000_lam",
        expert_notes="Laminar axisymmetric pipe Re=1000. Validate parabolic profile.",
    ),
    PromptCase(
        prompt="2D axisymmetric laminar pipe, Re=2000, D=0.02m, L=0.3m, air, wedge mesh",
        params=_wedge(2000, 0.02, 0.3),
        case_tag="wedge_re2000_lam",
        expert_notes="Approaching transition but still laminar. Wedge BC on front/back.",
    ),
    PromptCase(
        prompt="turbulent axisymmetric pipe flow with wedge: Re=10000, diameter=5cm, length=0.5m",
        params=_wedge(10000, 0.05, 0.5),
        case_tag="wedge_re10k_turb",
        expert_notes="Turbulent axisymmetric pipe. kOmegaSST with wall functions.",
    ),
    PromptCase(
        prompt="fully turbulent wedge pipe simulation: Re=50000, D=5cm, L=50cm, kOmegaSST",
        params=_wedge(50000, 0.05, 0.5),
        case_tag="wedge_re50k_turb",
        expert_notes="High Re turbulent axisymmetric pipe. y+ target 30.",
    ),
    PromptCase(
        prompt="axisymmetric high-Re pipe Re=100000, D=5cm, L=50cm, turbulent, wedge BC",
        params=_wedge(100000, 0.05, 0.5),
        case_tag="wedge_re100k_turb",
        expert_notes="Very high Re turbulent pipe. kOmegaSST wall functions critical.",
    ),
    PromptCase(
        prompt="axisymmetric laminar water pipe, Re=500, D=20mm, L=300mm, wedge geometry",
        params=_wedge(500, 0.02, 0.3, fluid="water"),
        case_tag="wedge_re500_water",
        expert_notes="Water properties nu=1e-6. Very low velocity U=0.025 m/s.",
    ),
    PromptCase(
        prompt="wedge axisymmetric pipe Re=20000, turbulent, D=0.05m, L=0.5m, air, RANS kOmegaSST",
        params=_wedge(20000, 0.05, 0.5),
        case_tag="wedge_re20k_turb",
        expert_notes="Moderately turbulent axisymmetric pipe.",
    ),

    # ── BOX (general duct / flat-plate) — simpleFoam ──────────────────────

    PromptCase(
        prompt="laminar rectangular duct flow, Re=1000, length=2m, height=0.1m, 2D, air",
        params=_box(1000, 2.0, 0.1),
        case_tag="box_re1k_lam",
        expert_notes="Laminar duct. Poiseuille-like profile. No turbulence model.",
    ),
    PromptCase(
        prompt="turbulent duct flow 2D, Re=10000, L=3m, H=0.1m, kOmegaSST, air",
        params=_box(10000, 3.0, 0.1),
        case_tag="box_re10k_turb",
        expert_notes="Turbulent duct. kOmegaSST wall functions on top/bottom walls.",
    ),
    PromptCase(
        prompt="low Reynolds number duct flow: Re=500, 2D box geometry, L=1m, H=0.05m, steady",
        params=_box(500, 1.0, 0.05),
        case_tag="box_re500_lam",
        expert_notes="Very laminar duct. Entrance length short.",
    ),
    PromptCase(
        prompt="2D turbulent channel in a box domain, Re=5000, L=2m, H=0.1m, air",
        params=_box(5000, 2.0, 0.1),
        case_tag="box_re5k_turb",
        expert_notes="Turbulent box duct flow. Re barely above transition.",
    ),
    PromptCase(
        prompt="high Re rectangular duct Re=20000, L=4m, H=0.15m, turbulent 2D air, simpleFoam",
        params=_box(20000, 4.0, 0.15),
        case_tag="box_re20k_turb",
        expert_notes="Strong turbulent duct. Large domain needed for full development.",
    ),
    PromptCase(
        prompt="flat plate flow in a box domain: Re=3000 (transitional), L=2m, H=0.1m, 2D",
        params=_box(3000, 2.0, 0.1),
        case_tag="box_re3k_trans",
        expert_notes="Transitional Re, use laminar solver as conservative choice.",
    ),

    # ── TRANSIENT — icoFoam (laminar, Re<2300) ────────────────────────────

    PromptCase(
        prompt="transient lid-driven cavity flow Re=100, impulsive start, 2D, icoFoam, air",
        params=_cavity_transient(100),
        case_tag="cavity_transient_re100",
        expert_notes="icoFoam transient cavity. Watch vortex develop from t=0.",
    ),
    PromptCase(
        prompt="time-dependent cavity flow, Re=400, 2D, unsteady simulation, air",
        params=_cavity_transient(400),
        case_tag="cavity_transient_re400",
        expert_notes="Unsteady cavity Re=400. icoFoam. Slower vortex evolution.",
    ),
    PromptCase(
        prompt="transient 2D cylinder wake Re=100, D=0.1m, unsteady vortex shedding, icoFoam",
        params=_cylinder_transient(100, 0.1),
        case_tag="cyl_transient_re100",
        expert_notes="Onset of vortex shedding. St ≈ 0.165. icoFoam laminar transient.",
    ),
    PromptCase(
        prompt="unsteady 2D flow past a cylinder, Re=200, D=0.1m, capture Karman vortex street",
        params=_cylinder_transient(200, 0.1),
        case_tag="cyl_transient_re200",
        expert_notes="Classic Karman street Re=200. icoFoam, Strouhal ≈ 0.19.",
    ),
    PromptCase(
        prompt="transient laminar pipe startup, Re=500, D=0.02m, L=0.3m, air, icoFoam",
        params=_pipe_transient(500, 0.02, 0.3),
        case_tag="pipe_transient_re500",
        expert_notes="Pipe flow startup transient. icoFoam, approaches Hagen-Poiseuille at steady state.",
    ),

    # ── TRANSIENT — pimpleFoam (turbulent or Re>=2300) ────────────────────

    PromptCase(
        prompt="transient turbulent vortex shedding from a cylinder, Re=3800, D=0.1m, pimpleFoam",
        params=_cylinder_transient(3800, 0.1),
        case_tag="cyl_transient_re3800",
        expert_notes="Turbulent transient cylinder. pimpleFoam kOmegaSST. CFL-based dt.",
    ),
    PromptCase(
        prompt="unsteady turbulent flow past cylinder Re=10000, D=0.1m, 2D, kOmegaSST pimpleFoam",
        params=_cylinder_transient(10000, 0.1),
        case_tag="cyl_transient_re10k",
        expert_notes="High Re transient cylinder. pimpleFoam, kOmegaSST wall functions.",
    ),
    PromptCase(
        prompt="transient turbulent pipe flow startup, Re=5000, D=0.05m, L=0.5m, pimpleFoam",
        params=_pipe_transient(5000, 0.05, 0.5),
        case_tag="pipe_transient_re5k",
        expert_notes="Turbulent pipe startup. pimpleFoam kOmegaSST. Short end_time=5s.",
    ),
    PromptCase(
        prompt="unsteady turbulent pipe flow, Re=20000, D=0.05m, L=0.5m, time-accurate simulation",
        params=_pipe_transient(20000, 0.05, 0.5),
        case_tag="pipe_transient_re20k",
        expert_notes="High Re transient pipe. pimpleFoam, adaptive dt for CFL.",
    ),
    PromptCase(
        prompt="vortex shedding behind a cylinder at Re=500, D=0.1m, 2D transient laminar, icoFoam",
        params=_cylinder_transient(500, 0.1),
        case_tag="cyl_transient_re500",
        expert_notes="Re=500 laminar transient → icoFoam (Re<2300, laminar).",
    ),

    # ── HEAT TRANSFER — buoyantSimpleFoam ─────────────────────────────────

    PromptCase(
        prompt="differentially heated square cavity, hot left wall, cold right wall, natural convection, air",
        params=_cavity_buoyancy(1.0),
        case_tag="buoy_cavity_diffheated",
        expert_notes="De Vahl Davis benchmark. buoyantSimpleFoam, gravity=-9.81 in y.",
    ),
    PromptCase(
        prompt="natural convection in a 1m square cavity with temperature gradient, buoyantSimpleFoam",
        params=_cavity_buoyancy(1.0),
        case_tag="buoy_cavity_natconv",
        expert_notes="Buoyancy-driven flow. No forced velocity. T gradient drives circulation.",
    ),
    PromptCase(
        prompt="buoyancy-driven flow simulation: square cavity 1m×1m, heated bottom wall, air",
        params=_cavity_buoyancy(1.0),
        case_tag="buoy_cavity_hotbottom",
        expert_notes="Rayleigh-Benard style. buoyantSimpleFoam, thermophysical properties.",
    ),
    PromptCase(
        prompt="thermal cavity CFD: 2D square box with hot wall at 350K and cold wall at 290K, air",
        params=_cavity_buoyancy(1.0),
        case_tag="buoy_cavity_tempbcs",
        expert_notes="Fixed temperature BCs. 0/T field: hot=350K, cold=290K, insulated top/bottom.",
    ),
    PromptCase(
        prompt="mixed convection in a square cavity: forced inflow and heated side wall, OpenFOAM buoyantSimpleFoam",
        params=_cavity_buoyancy(1.0),
        case_tag="buoy_cavity_mixed",
        expert_notes="Mixed convection. buoyantSimpleFoam with small inlet velocity for forced component.",
    ),

    # ── COMPRESSIBLE — rhoSimpleFoam (steady) ─────────────────────────────

    PromptCase(
        prompt="steady compressible channel flow at Mach 0.5, L=2m, H=0.1m, air, rhoSimpleFoam",
        params=_compressible_box(0.5, 2.0, 0.1, is_transient=False),
        case_tag="comp_box_ma05_steady",
        expert_notes="Subsonic compressible. rhoSimpleFoam, perfectGas thermo, p in Pa.",
    ),
    PromptCase(
        prompt="subsonic compressible duct flow, Mach=0.3, H=0.1m, L=2m, air, steady RANS",
        params=_compressible_box(0.3, 2.0, 0.1, is_transient=False),
        case_tag="comp_box_ma03_steady",
        expert_notes="Low subsonic compressible. rhoSimpleFoam, kOmegaSST.",
    ),
    PromptCase(
        prompt="high-speed pipe flow Ma=0.5, D=0.05m, L=0.5m, air, steady compressible CFD",
        params=_compressible_pipe(0.5, 0.05, 0.5, is_transient=False),
        case_tag="comp_pipe_ma05_steady",
        expert_notes="Compressible pipe rhoSimpleFoam. Outlet pressure fixed at 101325 Pa.",
    ),
    PromptCase(
        prompt="compressible internal flow in a duct: inlet Ma=0.8, transonic, H=0.1m, L=2m",
        params=_compressible_box(0.8, 2.0, 0.1, is_transient=False),
        case_tag="comp_box_ma08_steady",
        expert_notes="Near-transonic compressible. rhoSimpleFoam, density effects important.",
    ),

    # ── COMPRESSIBLE — rhoPimpleFoam (transient) ──────────────────────────

    PromptCase(
        prompt="transient compressible channel flow, Mach 0.5, L=2m, H=0.1m, air, rhoPimpleFoam",
        params=_compressible_box(0.5, 2.0, 0.1, is_transient=True),
        case_tag="comp_box_ma05_transient",
        expert_notes="Transient compressible. rhoPimpleFoam, PIMPLE loop, CFL-limited dt.",
    ),
    PromptCase(
        prompt="unsteady compressible pipe flow startup, Ma=0.3, D=0.05m, L=0.5m, air",
        params=_compressible_pipe(0.3, 0.05, 0.5, is_transient=True),
        case_tag="comp_pipe_ma03_transient",
        expert_notes="Transient compressible pipe. rhoPimpleFoam. Density wave propagation.",
    ),
    PromptCase(
        prompt="time-accurate compressible flow in a box: Ma=0.5, transient, air, kOmegaSST RANS",
        params=_compressible_box(0.5, 2.0, 0.1, is_transient=True),
        case_tag="comp_box_ma05_transient_b",
        expert_notes="rhoPimpleFoam with kOmegaSST. Energy equation with sensibleEnthalpy.",
    ),

    # ── MULTIPHASE — interFoam (VOF) ──────────────────────────────────────

    PromptCase(
        prompt="dam break simulation: water column collapses in air, 2D box 4m×2m, interFoam",
        params=_dam_break(4.0, 2.0),
        case_tag="multiphase_dambreak",
        expert_notes="Classic VOF dam break. interFoam, alpha.water VOF field, gravity -9.81.",
    ),
    PromptCase(
        prompt="2D dam break: 1m water column in a 4m × 2m domain, water and air, OpenFOAM VOF",
        params=_dam_break(4.0, 2.0),
        case_tag="multiphase_dambreak_b",
        expert_notes="Same dam break, beginner phrasing. interFoam, end_time=1s.",
    ),
    PromptCase(
        prompt="OpenFOAM interFoam: free surface wave in a channel, L=10m, H=2m, water-air interface",
        params=_wave_channel(10.0, 2.0),
        case_tag="multiphase_wave_channel",
        expert_notes="Free surface channel. interFoam VOF, sigma=0.07 N/m, gravity.",
    ),
    PromptCase(
        prompt="water-air two-phase flow in a rectangular domain, dam break, transient VOF simulation",
        params=_dam_break(4.0, 2.0),
        case_tag="multiphase_dambreak_c",
        expert_notes="Two-phase VOF. interFoam with two-phase transportProperties.",
    ),
    PromptCase(
        prompt="free surface flow simulation: wave tank 10m×2m, water and air, unsteady 2D",
        params=_wave_channel(10.0, 2.0),
        case_tag="multiphase_wavetank",
        expert_notes="Wave tank VOF. interFoam, Courant-limited dt for interface tracking.",
    ),

    # ── buoyantSimpleFoam augmented ────────────────────────────────────────

    PromptCase(
        prompt="natural convection in a small 0.5m × 0.5m heated cavity, air, buoyantSimpleFoam",
        params=_buoyancy(0.5),
        case_tag="buoy_cavity_05m",
        expert_notes="Laminar natural convection. Small cavity, lower Ra.",
    ),
    PromptCase(
        prompt="large differentially heated cavity 2m × 2m, hot left cold right, natural convection air",
        params=_buoyancy(2.0),
        case_tag="buoy_cavity_2m",
        expert_notes="Larger cavity, higher Ra. Still laminar buoyantSimpleFoam.",
    ),
    PromptCase(
        prompt="high Rayleigh number natural convection in a 1m square cavity, turbulent, buoyantSimpleFoam kOmegaSST",
        params=_buoyancy(1.0, turb=True),
        case_tag="buoy_cavity_turb_1m",
        expert_notes="High Ra turbulent natural convection. kOmegaSST with buoyantSimpleFoam.",
    ),
    PromptCase(
        prompt="turbulent natural convection in a 2m × 2m differentially heated enclosure, air, Ra > 1e9",
        params=_buoyancy(2.0, turb=True),
        case_tag="buoy_cavity_turb_2m",
        expert_notes="High Ra large cavity. Turbulent buoyantSimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="OpenFOAM buoyantSimpleFoam: room-scale natural convection 3m × 2m cavity, hot wall 350K cold 290K",
        params=_buoyancy(3.0, turb=True, end_time=3000.0),
        case_tag="buoy_room_3x2_turb",
        expert_notes="Room-scale turbulent natural convection. Ra very high, kOmegaSST.",
    ),
    PromptCase(
        prompt="buoyancy-driven air flow in a 0.25m × 0.25m micro-cavity, heated side wall, laminar",
        params=_buoyancy(0.25),
        case_tag="buoy_cavity_025m",
        expert_notes="Very small cavity, low Ra, laminar natural convection.",
    ),
    PromptCase(
        prompt="thermal stratification in a 4m × 3m building room: hot floor 310K, cold ceiling 290K, air",
        params=_buoyancy(4.0, turb=True, end_time=3000.0),
        case_tag="buoy_building_4x3",
        expert_notes="Building room scale. Turbulent buoyantSimpleFoam.",
    ),
    PromptCase(
        prompt="natural convection benchmark: De Vahl Davis cavity, side-wall temperature difference, 1m square, air",
        params=_buoyancy(1.0),
        case_tag="buoy_devahldavis",
        expert_notes="De Vahl Davis benchmark. Laminar, Ra=10^4-10^6 range.",
    ),
    PromptCase(
        prompt="heated cavity simulation: 1.5m × 1.5m square domain, hot left wall 350K, cold right 300K, steady-state",
        params=_buoyancy(1.5),
        case_tag="buoy_cavity_15m",
        expert_notes="Intermediate size laminar cavity. buoyantSimpleFoam steady.",
    ),
    PromptCase(
        prompt="simulate natural convection in a vertical channel with hot and cold walls, buoyantSimpleFoam, turbulent",
        params=_buoyancy(1.0, turb=True),
        case_tag="buoy_cavity_turb_1m_b",
        expert_notes="Turbulent natural convection vertical channel approximation.",
    ),
    PromptCase(
        prompt="Rayleigh-Bénard convection: heated bottom wall 360K, cooled top 300K, 2D square cavity 1m, air",
        params=_buoyancy(1.0),
        case_tag="buoy_rayleigh_benard",
        expert_notes="Rayleigh-Bénard. buoyantSimpleFoam, gravity in -y direction.",
    ),
    PromptCase(
        prompt="CFD natural convection study: differentially heated enclosure, H=0.5m W=1m, air, steady heat transfer",
        params=_buoyancy(0.5),
        case_tag="buoy_rect_half",
        expert_notes="Aspect ratio 2:1 cavity. buoyantSimpleFoam laminar.",
    ),

    # ── icoFoam augmented ─────────────────────────────────────────────────

    PromptCase(
        prompt="transient lid-driven cavity Re=200, 0.5m × 0.5m, impulsive start, 2D air, icoFoam",
        params=_cavity_transient(200, 0.5),
        case_tag="ico_cavity_re200_05m",
        expert_notes="icoFoam laminar transient. Small cavity Re=200.",
    ),
    PromptCase(
        prompt="unsteady cavity flow Re=1000, 2D square 1m, laminar, icoFoam, air",
        params=_cavity_transient(1000),
        case_tag="ico_cavity_re1000",
        expert_notes="icoFoam Re=1000 — approaching transitional but classified laminar.",
    ),
    PromptCase(
        prompt="time-dependent lid-driven cavity, Re=100, 2m × 2m domain, laminar air, icoFoam",
        params=_cavity_transient(100, 2.0),
        case_tag="ico_cavity_re100_2m",
        expert_notes="Large cavity Re=100. Very slow lid velocity. icoFoam.",
    ),
    PromptCase(
        prompt="laminar transient flow past a cylinder Re=50, D=0.1m, 2D unsteady air, icoFoam",
        params=_cylinder_transient(50, 0.1),
        case_tag="ico_cyl_re50",
        expert_notes="Very low Re laminar cylinder. Fully attached, no shedding. icoFoam.",
    ),
    PromptCase(
        prompt="transient laminar cylinder flow Re=150, D=0.05m, 2D, icoFoam, capture wake dynamics",
        params=_cylinder_transient(150, 0.05),
        case_tag="ico_cyl_re150_d005",
        expert_notes="Re=150 near onset of vortex shedding (Re_crit≈47). icoFoam.",
    ),
    PromptCase(
        prompt="unsteady laminar flow around a cylinder Re=300, D=0.1m, 2D air, icoFoam",
        params=_cylinder_transient(300, 0.1),
        case_tag="ico_cyl_re300",
        expert_notes="Re=300 laminar (Re<2300). Vortex shedding active. icoFoam.",
    ),
    PromptCase(
        prompt="transient laminar pipe startup flow, Re=1000, D=2cm, L=30cm, air, icoFoam",
        params=_pipe_transient(1000, 0.02, 0.3),
        case_tag="ico_pipe_re1000",
        expert_notes="Laminar pipe startup. Womersley/impulse start. icoFoam.",
    ),
    PromptCase(
        prompt="unsteady laminar pipe flow Re=2000, D=0.02m, L=0.3m, approaching transition, icoFoam",
        params=_pipe_transient(2000, 0.02, 0.3),
        case_tag="ico_pipe_re2000",
        expert_notes="Re=2000 laminar (below 2300 threshold). icoFoam transient pipe.",
    ),
    PromptCase(
        prompt="transient laminar channel flow Re=500, L=5m, H=0.1m, 2D air, icoFoam",
        params=_channel_transient(500, 5.0, 0.1),
        case_tag="ico_chan_re500",
        expert_notes="Laminar transient channel. icoFoam, Poiseuille profile develops.",
    ),
    PromptCase(
        prompt="unsteady laminar channel flow Re=1000, L=5m, H=0.1m, time-accurate simulation, air",
        params=_channel_transient(1000, 5.0, 0.1),
        case_tag="ico_chan_re1000",
        expert_notes="Laminar transient channel Re=1000. icoFoam.",
    ),
    PromptCase(
        prompt="transient laminar backward-facing step Re=400, step height=0.1m, 2D air, icoFoam",
        params=_bfs_transient(400, 0.1),
        case_tag="ico_bfs_re400",
        expert_notes="Laminar transient BFS. Reattachment oscillates before reaching steady state.",
    ),
    PromptCase(
        prompt="unsteady 2D BFS flow Re=800, h=0.1m, laminar icoFoam, air, transient simulation",
        params=_bfs_transient(800, 0.1),
        case_tag="ico_bfs_re800",
        expert_notes="Transient BFS Re=800. icoFoam laminar. Captures development of recirculation.",
    ),

    # ── pimpleFoam augmented ───────────────────────────────────────────────

    PromptCase(
        prompt="turbulent transient vortex shedding cylinder Re=5000, D=0.1m, 2D, pimpleFoam kOmegaSST",
        params=_cylinder_transient(5000, 0.1),
        case_tag="pimple_cyl_re5k",
        expert_notes="Turbulent transient cylinder Re=5000. pimpleFoam kOmegaSST, CFL-limited dt.",
    ),
    PromptCase(
        prompt="high-Re transient wake behind a cylinder, Re=20000, D=0.1m, 2D, unsteady turbulent CFD",
        params=_cylinder_transient(20000, 0.1),
        case_tag="pimple_cyl_re20k",
        expert_notes="High Re turbulent transient cylinder. pimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="transient turbulent flow over a cylinder at Re=50000, D=0.1m, 2D pimpleFoam RANS",
        params=_cylinder_transient(50000, 0.1),
        case_tag="pimple_cyl_re50k",
        expert_notes="Very high Re turbulent cylinder. pimpleFoam kOmegaSST wall functions.",
    ),
    PromptCase(
        prompt="transient turbulent cylinder Re=2500, D=0.1m, 2D, pimpleFoam, unsteady wake",
        params=_cylinder_transient(2500, 0.1),
        case_tag="pimple_cyl_re2500",
        expert_notes="Just above turbulent threshold Re=2500. pimpleFoam.",
    ),
    PromptCase(
        prompt="turbulent transient cylinder Re=3000, D=0.05m, 2D air, pimpleFoam kOmegaSST",
        params=_cylinder_transient(3000, 0.05),
        case_tag="pimple_cyl_re3k_d005",
        expert_notes="Turbulent cylinder smaller diameter D=0.05m. pimpleFoam.",
    ),
    PromptCase(
        prompt="unsteady turbulent channel flow Re=5000, L=5m, H=0.1m, 2D air, pimpleFoam",
        params=_channel_transient(5000, 5.0, 0.1),
        case_tag="pimple_chan_re5k",
        expert_notes="Turbulent transient channel Re=5000. pimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="transient turbulent channel flow Re=10000, length 5m, height 0.1m, time-accurate RANS",
        params=_channel_transient(10000, 5.0, 0.1),
        case_tag="pimple_chan_re10k",
        expert_notes="High Re transient channel. pimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="unsteady turbulent channel Re=20000, L=4m, H=0.2m, 2D air, pimpleFoam",
        params=_channel_transient(20000, 4.0, 0.2),
        case_tag="pimple_chan_re20k",
        expert_notes="Very high Re transient channel. Strong wall shear, pimpleFoam.",
    ),
    PromptCase(
        prompt="transient turbulent backward-facing step Re=5000, h=0.1m, 2D air, pimpleFoam",
        params=_bfs_transient(5000, 0.1),
        case_tag="pimple_bfs_re5k",
        expert_notes="Turbulent transient BFS. pimpleFoam kOmegaSST. Unsteady reattachment.",
    ),
    PromptCase(
        prompt="unsteady high-Re backward step flow, Re=10000, h=0.1m, 2D turbulent, pimpleFoam kOmegaSST",
        params=_bfs_transient(10000, 0.1),
        case_tag="pimple_bfs_re10k",
        expert_notes="High Re turbulent transient BFS. Large recirculation zone dynamics.",
    ),
    PromptCase(
        prompt="turbulent transient pipe flow Re=10000, D=0.05m, L=0.5m, unsteady 3D, pimpleFoam",
        params=_pipe_transient(10000, 0.05, 0.5),
        case_tag="pimple_pipe_re10k",
        expert_notes="Turbulent transient pipe Re=10000. pimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="high-Re turbulent pipe startup, Re=50000, D=5cm, L=0.5m, transient 3D, pimpleFoam",
        params=_pipe_transient(50000, 0.05, 0.5),
        case_tag="pimple_pipe_re50k",
        expert_notes="Very high Re turbulent pipe transient. pimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="transient turbulent pipe flow Re=3000, D=0.05m, L=0.5m, air, pimpleFoam",
        params=_pipe_transient(3000, 0.05, 0.5),
        case_tag="pimple_pipe_re3k",
        expert_notes="Re=3000 turbulent pipe startup. pimpleFoam.",
    ),
    PromptCase(
        prompt="unsteady turbulent channel flow Re=3000, L=5m, H=0.1m, 2D air, pimpleFoam RANS",
        params=_channel_transient(3000, 5.0, 0.1),
        case_tag="pimple_chan_re3k",
        expert_notes="Re=3000 transient turbulent channel. pimpleFoam kOmegaSST.",
    ),

    # ── rhoSimpleFoam augmented ────────────────────────────────────────────

    PromptCase(
        prompt="steady compressible duct flow Mach 0.4, L=2m, H=0.1m, air, rhoSimpleFoam",
        params=_compressible_box(0.4, 2.0, 0.1, is_transient=False),
        case_tag="rhoSimple_box_ma04",
        expert_notes="Subsonic compressible Ma=0.4. rhoSimpleFoam, perfectGas thermo.",
    ),
    PromptCase(
        prompt="compressible channel flow at Mach 0.6, L=2m, H=0.1m, steady-state, air",
        params=_compressible_box(0.6, 2.0, 0.1, is_transient=False),
        case_tag="rhoSimple_box_ma06",
        expert_notes="Moderate subsonic compressible Ma=0.6. Density variation ~20%.",
    ),
    PromptCase(
        prompt="high-speed duct flow Mach 0.7, 2D, L=2m, H=0.1m, air, steady RANS rhoSimpleFoam",
        params=_compressible_box(0.7, 2.0, 0.1, is_transient=False),
        case_tag="rhoSimple_box_ma07",
        expert_notes="High subsonic Ma=0.7. Strong compressibility effects. rhoSimpleFoam.",
    ),
    PromptCase(
        prompt="near-transonic duct, Mach 0.8, L=4m, H=0.1m, air, steady compressible CFD",
        params=_compressible_box(0.8, 4.0, 0.1, is_transient=False),
        case_tag="rhoSimple_box_ma08_long",
        expert_notes="Near-transonic Ma=0.8. rhoSimpleFoam, kOmegaSST.",
    ),
    PromptCase(
        prompt="steady compressible pipe flow Ma=0.3, D=2cm, L=0.3m, air, rhoSimpleFoam",
        params=_compressible_pipe(0.3, 0.02, 0.3, is_transient=False),
        case_tag="rhoSimple_pipe_ma03_small",
        expert_notes="Small compressible pipe Ma=0.3. rhoSimpleFoam, outlet pressure 101325 Pa.",
    ),
    PromptCase(
        prompt="compressible pipe flow at Mach 0.4, D=5cm, L=0.5m, steady, air, rhoSimpleFoam",
        params=_compressible_pipe(0.4, 0.05, 0.5, is_transient=False),
        case_tag="rhoSimple_pipe_ma04",
        expert_notes="Subsonic compressible pipe Ma=0.4. rhoSimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="high-speed pipe flow, Mach 0.6, D=5cm, L=50cm, steady compressible, air",
        params=_compressible_pipe(0.6, 0.05, 0.5, is_transient=False),
        case_tag="rhoSimple_pipe_ma06",
        expert_notes="Compressible pipe Ma=0.6. rhoSimpleFoam.",
    ),
    PromptCase(
        prompt="transonic pipe flow Ma=0.8, D=5cm, L=0.5m, steady-state, air, rhoSimpleFoam",
        params=_compressible_pipe(0.8, 0.05, 0.5, is_transient=False),
        case_tag="rhoSimple_pipe_ma08",
        expert_notes="Near-transonic pipe Ma=0.8. rhoSimpleFoam density-based solver.",
    ),
    PromptCase(
        prompt="steady subsonic compressible channel Ma=0.5, larger domain L=3m H=0.2m, air, RANS",
        params=_compressible_box(0.5, 3.0, 0.2, is_transient=False),
        case_tag="rhoSimple_box_ma05_large",
        expert_notes="Compressible Ma=0.5 in wider channel. rhoSimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="compressible duct flow Ma=0.3, small channel L=1m H=0.05m, steady, air, rhoSimpleFoam",
        params=_compressible_box(0.3, 1.0, 0.05, is_transient=False),
        case_tag="rhoSimple_box_ma03_small",
        expert_notes="Small duct compressible Ma=0.3. Low compressibility effects.",
    ),
    PromptCase(
        prompt="compressible pipe flow Ma=0.7, D=5cm, L=0.5m, turbulent air, steady rhoSimpleFoam kOmegaSST",
        params=_compressible_pipe(0.7, 0.05, 0.5, is_transient=False),
        case_tag="rhoSimple_pipe_ma07",
        expert_notes="Turbulent compressible pipe Ma=0.7.",
    ),
    PromptCase(
        prompt="steady-state compressible channel, Mach 0.5, wider domain L=2m H=0.2m, air",
        params=_compressible_box(0.5, 2.0, 0.2, is_transient=False),
        case_tag="rhoSimple_box_ma05_wide",
        expert_notes="Compressible wide channel Ma=0.5. rhoSimpleFoam.",
    ),

    # ── rhoPimpleFoam augmented ────────────────────────────────────────────

    PromptCase(
        prompt="transient compressible duct flow Ma=0.3, L=2m, H=0.1m, air, rhoPimpleFoam",
        params=_compressible_box(0.3, 2.0, 0.1, is_transient=True),
        case_tag="rhoPimple_box_ma03",
        expert_notes="Low subsonic transient compressible. rhoPimpleFoam.",
    ),
    PromptCase(
        prompt="unsteady compressible channel flow at Mach 0.4, L=2m, H=0.1m, air, rhoPimpleFoam",
        params=_compressible_box(0.4, 2.0, 0.1, is_transient=True),
        case_tag="rhoPimple_box_ma04",
        expert_notes="Transient compressible Ma=0.4. rhoPimpleFoam PIMPLE loop.",
    ),
    PromptCase(
        prompt="transient high-speed channel Mach 0.6, L=2m, H=0.1m, air, time-accurate rhoPimpleFoam",
        params=_compressible_box(0.6, 2.0, 0.1, is_transient=True),
        case_tag="rhoPimple_box_ma06",
        expert_notes="Transient compressible Ma=0.6. rhoPimpleFoam.",
    ),
    PromptCase(
        prompt="transient near-transonic duct Ma=0.7, L=2m, H=0.1m, air, unsteady rhoPimpleFoam",
        params=_compressible_box(0.7, 2.0, 0.1, is_transient=True),
        case_tag="rhoPimple_box_ma07",
        expert_notes="Near-transonic transient. rhoPimpleFoam energy equation.",
    ),
    PromptCase(
        prompt="unsteady compressible pipe startup Ma=0.4, D=5cm, L=0.5m, air, rhoPimpleFoam",
        params=_compressible_pipe(0.4, 0.05, 0.5, is_transient=True),
        case_tag="rhoPimple_pipe_ma04",
        expert_notes="Transient compressible pipe Ma=0.4. rhoPimpleFoam.",
    ),
    PromptCase(
        prompt="transient compressible pipe flow Ma=0.5, D=5cm, L=0.5m, air, unsteady RANS",
        params=_compressible_pipe(0.5, 0.05, 0.5, is_transient=True),
        case_tag="rhoPimple_pipe_ma05",
        expert_notes="Compressible pipe transient Ma=0.5. rhoPimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="high-speed transient pipe flow Mach 0.6, D=5cm, L=0.5m, air, rhoPimpleFoam",
        params=_compressible_pipe(0.6, 0.05, 0.5, is_transient=True),
        case_tag="rhoPimple_pipe_ma06",
        expert_notes="Transient compressible pipe Ma=0.6. Density wave propagation.",
    ),
    PromptCase(
        prompt="transient compressible small duct Ma=0.3, L=1m, H=0.05m, air, rhoPimpleFoam",
        params=_compressible_box(0.3, 1.0, 0.05, is_transient=True),
        case_tag="rhoPimple_box_ma03_small",
        expert_notes="Small transient compressible duct. Low Ma, mild density variation.",
    ),
    PromptCase(
        prompt="unsteady compressible wider channel Ma=0.5, L=3m, H=0.2m, air, rhoPimpleFoam",
        params=_compressible_box(0.5, 3.0, 0.2, is_transient=True),
        case_tag="rhoPimple_box_ma05_wide",
        expert_notes="Wider transient compressible channel Ma=0.5. rhoPimpleFoam.",
    ),
    PromptCase(
        prompt="transient compressible small pipe startup, Ma=0.3, D=2cm, L=30cm, air, rhoPimpleFoam",
        params=_compressible_pipe(0.3, 0.02, 0.3, is_transient=True),
        case_tag="rhoPimple_pipe_ma03_small",
        expert_notes="Small transient compressible pipe. rhoPimpleFoam acoustic startup.",
    ),
    PromptCase(
        prompt="unsteady turbulent compressible pipe flow, Mach=0.7, D=5cm, L=0.5m, air, rhoPimpleFoam",
        params=_compressible_pipe(0.7, 0.05, 0.5, is_transient=True),
        case_tag="rhoPimple_pipe_ma07",
        expert_notes="High-speed transient pipe Ma=0.7. rhoPimpleFoam kOmegaSST.",
    ),
    PromptCase(
        prompt="time-accurate compressible long duct Ma=0.4, L=4m, H=0.15m, air, rhoPimpleFoam RANS",
        params=_compressible_box(0.4, 4.0, 0.15, is_transient=True),
        case_tag="rhoPimple_box_ma04_long",
        expert_notes="Long transient compressible duct. rhoPimpleFoam.",
    ),

    # ── interFoam augmented ────────────────────────────────────────────────

    PromptCase(
        prompt="small dam break: 1m water column in a 2m × 1m box, water-air, VOF interFoam",
        params=_dam_break(2.0, 1.0),
        case_tag="multiphase_dambreak_small",
        expert_notes="Small scale dam break. interFoam VOF. Fast collapse dynamics.",
    ),
    PromptCase(
        prompt="large-scale dam break simulation: 8m × 4m domain, water and air, interFoam 2D",
        params=_dam_break(8.0, 4.0),
        case_tag="multiphase_dambreak_large",
        expert_notes="Large dam break. interFoam, gravity=-9.81, sigma=0.07 N/m.",
    ),
    PromptCase(
        prompt="tall water column collapse: 3m wide × 2m high domain, water dam break, VOF simulation",
        params=_dam_break(3.0, 2.0),
        case_tag="multiphase_dambreak_tall",
        expert_notes="Tall water column dam break. interFoam, high Re VOF.",
    ),
    PromptCase(
        prompt="large dam break: 4m wide 3m tall domain, initial 2m water column, interFoam VOF",
        params=_dam_break(4.0, 3.0),
        case_tag="multiphase_dambreak_wide",
        expert_notes="Wide tall dam break. interFoam, alpha.water=1 in initial column.",
    ),
    PromptCase(
        prompt="free surface wave propagation in a 5m × 1m tank, water-air interface, interFoam",
        params=_wave_channel(5.0, 1.0),
        case_tag="multiphase_wave_5x1",
        expert_notes="Short wave tank. interFoam VOF, surface tension 0.07 N/m.",
    ),
    PromptCase(
        prompt="OpenFOAM wave tank simulation: 15m long, 3m deep, water and air, free surface tracking",
        params=_wave_channel(15.0, 3.0),
        case_tag="multiphase_wave_15x3",
        expert_notes="Large wave tank. interFoam VOF, Courant-limited dt for interface.",
    ),
    PromptCase(
        prompt="free surface channel flow: 8m × 2m, water and air, VOF simulation, unsteady",
        params=_wave_channel(8.0, 2.0),
        case_tag="multiphase_wave_8x2",
        expert_notes="Medium wave channel. interFoam.",
    ),
    PromptCase(
        prompt="sloshing tank simulation: 2m × 1m partially filled tank, water-air VOF, interFoam",
        params=_sloshing(2.0, 1.0),
        case_tag="multiphase_slosh_2x1",
        expert_notes="Sloshing tank VOF. interFoam, gravity-driven interface oscillation.",
    ),
    PromptCase(
        prompt="liquid sloshing in a 3m × 1.5m container, water and air, transient VOF interFoam",
        params=_sloshing(3.0, 1.5),
        case_tag="multiphase_slosh_3x15",
        expert_notes="Larger sloshing tank. interFoam VOF, natural frequency of sloshing.",
    ),
    PromptCase(
        prompt="small sloshing tank 1m × 0.5m, water and air two-phase flow, interFoam VOF",
        params=_sloshing(1.0, 0.5),
        case_tag="multiphase_slosh_1x05",
        expert_notes="Small sloshing container. interFoam, fast dynamics.",
    ),
    PromptCase(
        prompt="large liquid sloshing tank: 4m wide, 2m deep, water-air interface, VOF simulation",
        params=_sloshing(4.0, 2.0),
        case_tag="multiphase_slosh_4x2",
        expert_notes="Large sloshing tank. interFoam, potential resonance effects.",
    ),
    PromptCase(
        prompt="two-phase free surface flow: rising water in a 2m × 2m box, interFoam, gravity effects",
        params=_dam_break(2.0, 2.0),
        case_tag="multiphase_risingwater",
        expert_notes="Water rising from partial dam break in square domain. interFoam VOF.",
    ),

    # ── icoFoam augmented v2 ──────────────────────────────────────────────
    PromptCase(prompt="2D laminar transient cavity Re=400, 1m × 1m, icoFoam air", params=_cavity_transient(400, 1.0), case_tag="ico_cav_re400", expert_notes="Re=400 transient lid-driven cavity icoFoam."),
    PromptCase(prompt="transient lid-driven cavity flow Re=600, 1m square, laminar air, icoFoam", params=_cavity_transient(600, 1.0), case_tag="ico_cav_re600", expert_notes="Re=600 transient cavity. icoFoam laminar."),
    PromptCase(prompt="time-dependent cavity flow Re=800, 0.75m × 0.75m, air, icoFoam laminar", params=_cavity_transient(800, 0.75), case_tag="ico_cav_re800_075", expert_notes="Re=800 cavity icoFoam."),
    PromptCase(prompt="impulsive lid cavity Re=1500, 1m × 1m, icoFoam, transient laminar", params=_cavity_transient(1500, 1.0), case_tag="ico_cav_re1500", expert_notes="Re=1500 cavity. icoFoam laminar."),
    PromptCase(prompt="transient cavity Re=2000, 1.5m × 1.5m air, icoFoam", params=_cavity_transient(2000, 1.5), case_tag="ico_cav_re2000_15", expert_notes="Re=2000 large cavity. icoFoam transient."),
    PromptCase(prompt="laminar transient flow past cylinder Re=80, D=0.05m, 2D, icoFoam", params=_cylinder_transient(80, 0.05), case_tag="ico_cyl_re80", expert_notes="Re=80 cylinder. Onset of vortex shedding. icoFoam."),
    PromptCase(prompt="cylinder vortex shedding Re=100, D=0.1m, transient laminar icoFoam, air", params=_cylinder_transient(100, 0.1), case_tag="ico_cyl_re100", expert_notes="Re=100 classical Karman vortex street. icoFoam."),
    PromptCase(prompt="cylinder Re=200 transient, D=0.05m, 2D laminar wake icoFoam", params=_cylinder_transient(200, 0.05), case_tag="ico_cyl_re200", expert_notes="Re=200 cylinder shedding. icoFoam laminar."),
    PromptCase(prompt="laminar transient cylinder flow Re=500, D=0.1m, 2D air icoFoam", params=_cylinder_transient(500, 0.1), case_tag="ico_cyl_re500", expert_notes="Re=500 cylinder. Strong vortex shedding. icoFoam laminar."),
    PromptCase(prompt="transient laminar pipe flow Re=500, D=0.02m, L=0.3m, icoFoam, air startup", params=_pipe_transient(500, 0.02, 0.3), case_tag="ico_pipe_re500", expert_notes="Low Re pipe startup. icoFoam laminar transient."),
    PromptCase(prompt="laminar pipe Re=1500, D=0.025m, L=0.4m, transient icoFoam", params=_pipe_transient(1500, 0.025, 0.4), case_tag="ico_pipe_re1500", expert_notes="Re=1500 transient pipe. icoFoam."),
    PromptCase(prompt="transient channel flow Re=200, L=4m H=0.1m, laminar 2D air, icoFoam", params=_channel_transient(200, 4.0, 0.1), case_tag="ico_chan_re200", expert_notes="Re=200 transient channel. icoFoam Poiseuille startup."),
    PromptCase(prompt="laminar channel Re=800, L=5m H=0.15m, 2D transient air icoFoam", params=_channel_transient(800, 5.0, 0.15), case_tag="ico_chan_re800", expert_notes="Re=800 channel transient. icoFoam laminar."),
    PromptCase(prompt="laminar channel Re=1500, L=6m H=0.1m transient icoFoam air", params=_channel_transient(1500, 6.0, 0.1), case_tag="ico_chan_re1500", expert_notes="Re=1500 channel transient. icoFoam."),
    PromptCase(prompt="laminar BFS Re=200, step h=0.05m, transient icoFoam", params=_bfs_transient(200, 0.05), case_tag="ico_bfs_re200", expert_notes="Re=200 BFS transient. icoFoam laminar."),
    PromptCase(prompt="transient laminar BFS Re=600, step=0.1m, icoFoam, air, 2D", params=_bfs_transient(600, 0.1), case_tag="ico_bfs_re600", expert_notes="Re=600 BFS transient. icoFoam."),
    PromptCase(prompt="laminar backward facing step Re=1000, h=0.15m, transient icoFoam", params=_bfs_transient(1000, 0.15), case_tag="ico_bfs_re1000", expert_notes="Re=1000 BFS transient. icoFoam laminar."),
    PromptCase(prompt="laminar BFS Re=1500, h=0.1m, transient air icoFoam 2D", params=_bfs_transient(1500, 0.1), case_tag="ico_bfs_re1500", expert_notes="Re=1500 BFS transient. icoFoam."),
    PromptCase(prompt="laminar transient lid cavity Re=300 small 0.4m square icoFoam", params=_cavity_transient(300, 0.4), case_tag="ico_cav_re300_04m", expert_notes="Re=300 small cavity icoFoam."),
    PromptCase(prompt="laminar transient cavity Re=1000 0.6m square air icoFoam", params=_cavity_transient(1000, 0.6), case_tag="ico_cav_re1000_06m", expert_notes="Re=1000 medium cavity icoFoam."),
    PromptCase(prompt="laminar transient cavity Re=500 0.8m air icoFoam 2D", params=_cavity_transient(500, 0.8), case_tag="ico_cav_re500_08m", expert_notes="Re=500 cavity icoFoam transient."),
    PromptCase(prompt="laminar transient cavity Re=200 0.3m air icoFoam", params=_cavity_transient(200, 0.3), case_tag="ico_cav_re200_03m", expert_notes="Re=200 small cavity icoFoam."),

    # ── rhoSimpleFoam augmented v2 ────────────────────────────────────────
    PromptCase(prompt="steady compressible duct Ma=0.2 box L=2m H=0.1m air rhoSimpleFoam", params=_compressible_box(0.2, 2.0, 0.1, is_transient=False), case_tag="rhoSimple_box_ma02", expert_notes="Subsonic Ma=0.2 box. rhoSimpleFoam steady."),
    PromptCase(prompt="steady compressible flow box Ma=0.25 L=3m H=0.15m, air rhoSimpleFoam", params=_compressible_box(0.25, 3.0, 0.15, is_transient=False), case_tag="rhoSimple_box_ma025_long", expert_notes="Ma=0.25 long box. rhoSimpleFoam."),
    PromptCase(prompt="compressible duct flow Ma=0.35 box 2.5m × 0.12m air steady rhoSimpleFoam", params=_compressible_box(0.35, 2.5, 0.12, is_transient=False), case_tag="rhoSimple_box_ma035", expert_notes="Ma=0.35 medium box. rhoSimpleFoam steady."),
    PromptCase(prompt="rhoSimpleFoam steady compressible box Ma=0.45 4m × 0.2m air", params=_compressible_box(0.45, 4.0, 0.2, is_transient=False), case_tag="rhoSimple_box_ma045_wide", expert_notes="Ma=0.45 wide box. rhoSimpleFoam."),
    PromptCase(prompt="steady subsonic box flow Ma=0.55, L=2m H=0.1m, air, rhoSimpleFoam", params=_compressible_box(0.55, 2.0, 0.1, is_transient=False), case_tag="rhoSimple_box_ma055", expert_notes="Ma=0.55 box. rhoSimpleFoam steady."),
    PromptCase(prompt="rhoSimpleFoam Ma=0.65 box steady compressible 2m × 0.08m", params=_compressible_box(0.65, 2.0, 0.08, is_transient=False), case_tag="rhoSimple_box_ma065", expert_notes="Ma=0.65 narrow box. rhoSimpleFoam."),
    PromptCase(prompt="steady compressible pipe Ma=0.2 D=0.05m L=0.5m air rhoSimpleFoam", params=_compressible_pipe(0.2, 0.05, 0.5, is_transient=False), case_tag="rhoSimple_pipe_ma02", expert_notes="Ma=0.2 pipe. rhoSimpleFoam steady."),
    PromptCase(prompt="rhoSimpleFoam compressible pipe Ma=0.25 D=0.04m L=0.4m steady", params=_compressible_pipe(0.25, 0.04, 0.4, is_transient=False), case_tag="rhoSimple_pipe_ma025", expert_notes="Ma=0.25 pipe rhoSimpleFoam."),
    PromptCase(prompt="steady subsonic pipe Ma=0.35 D=0.05m L=0.5m air rhoSimpleFoam", params=_compressible_pipe(0.35, 0.05, 0.5, is_transient=False), case_tag="rhoSimple_pipe_ma035", expert_notes="Ma=0.35 pipe rhoSimpleFoam."),
    PromptCase(prompt="rhoSimpleFoam pipe Ma=0.45 D=0.06m L=0.6m steady compressible", params=_compressible_pipe(0.45, 0.06, 0.6, is_transient=False), case_tag="rhoSimple_pipe_ma045", expert_notes="Ma=0.45 pipe rhoSimpleFoam."),
    PromptCase(prompt="steady compressible pipe flow Ma=0.5 D=0.05m L=0.5m rhoSimpleFoam", params=_compressible_pipe(0.5, 0.05, 0.5, is_transient=False), case_tag="rhoSimple_pipe_ma05", expert_notes="Ma=0.5 pipe rhoSimpleFoam."),
    PromptCase(prompt="rhoSimpleFoam Ma=0.55 D=0.04m L=0.5m pipe air steady", params=_compressible_pipe(0.55, 0.04, 0.5, is_transient=False), case_tag="rhoSimple_pipe_ma055", expert_notes="Ma=0.55 pipe rhoSimpleFoam."),
    PromptCase(prompt="steady compressible pipe Ma=0.65, D=0.05m L=0.5m air rhoSimpleFoam", params=_compressible_pipe(0.65, 0.05, 0.5, is_transient=False), case_tag="rhoSimple_pipe_ma065", expert_notes="Ma=0.65 pipe rhoSimpleFoam."),
    PromptCase(prompt="compressible duct Ma=0.3 box 1.5m × 0.06m steady rhoSimpleFoam", params=_compressible_box(0.3, 1.5, 0.06, is_transient=False), case_tag="rhoSimple_box_ma03_narrow", expert_notes="Ma=0.3 narrow box. rhoSimpleFoam."),
    PromptCase(prompt="steady box Ma=0.4 long 5m × 0.18m air rhoSimpleFoam", params=_compressible_box(0.4, 5.0, 0.18, is_transient=False), case_tag="rhoSimple_box_ma04_xl", expert_notes="Ma=0.4 extra long box. rhoSimpleFoam."),
    PromptCase(prompt="rhoSimpleFoam steady compressible box Ma=0.5 3m × 0.1m air", params=_compressible_box(0.5, 3.0, 0.1, is_transient=False), case_tag="rhoSimple_box_ma05_med", expert_notes="Ma=0.5 medium box. rhoSimpleFoam."),
    PromptCase(prompt="steady subsonic compressible flow Ma=0.6 box 2m × 0.05m rhoSimpleFoam", params=_compressible_box(0.6, 2.0, 0.05, is_transient=False), case_tag="rhoSimple_box_ma06_narrow", expert_notes="Ma=0.6 narrow box. rhoSimpleFoam."),

    # ── rhoPimpleFoam augmented v2 ────────────────────────────────────────
    PromptCase(prompt="transient compressible box Ma=0.3 1.5m × 0.08m rhoPimpleFoam air", params=_compressible_box(0.3, 1.5, 0.08, is_transient=True), case_tag="rhoPimple_box_ma03_narrow", expert_notes="Ma=0.3 narrow transient box. rhoPimpleFoam."),
    PromptCase(prompt="rhoPimpleFoam transient box Ma=0.35 2m × 0.1m air", params=_compressible_box(0.35, 2.0, 0.1, is_transient=True), case_tag="rhoPimple_box_ma035", expert_notes="Ma=0.35 transient box rhoPimpleFoam."),
    PromptCase(prompt="transient compressible flow box Ma=0.4 long 3m × 0.12m rhoPimpleFoam", params=_compressible_box(0.4, 3.0, 0.12, is_transient=True), case_tag="rhoPimple_box_ma04_med", expert_notes="Ma=0.4 medium box transient rhoPimpleFoam."),
    PromptCase(prompt="rhoPimpleFoam Ma=0.45 box 2m × 0.1m transient air", params=_compressible_box(0.45, 2.0, 0.1, is_transient=True), case_tag="rhoPimple_box_ma045", expert_notes="Ma=0.45 transient box rhoPimpleFoam."),
    PromptCase(prompt="transient compressible box Ma=0.5, 2.5m × 0.1m, air rhoPimpleFoam", params=_compressible_box(0.5, 2.5, 0.1, is_transient=True), case_tag="rhoPimple_box_ma05", expert_notes="Ma=0.5 transient box rhoPimpleFoam."),
    PromptCase(prompt="rhoPimpleFoam compressible box Ma=0.55 wide 4m × 0.18m air", params=_compressible_box(0.55, 4.0, 0.18, is_transient=True), case_tag="rhoPimple_box_ma055_wide", expert_notes="Ma=0.55 wide transient box rhoPimpleFoam."),
    PromptCase(prompt="transient box Ma=0.4 short 1m × 0.06m rhoPimpleFoam air", params=_compressible_box(0.4, 1.0, 0.06, is_transient=True), case_tag="rhoPimple_box_ma04_short", expert_notes="Ma=0.4 short box rhoPimpleFoam transient."),
    PromptCase(prompt="rhoPimpleFoam transient pipe Ma=0.3 D=0.05m L=0.5m air", params=_compressible_pipe(0.3, 0.05, 0.5, is_transient=True), case_tag="rhoPimple_pipe_ma03", expert_notes="Ma=0.3 transient pipe rhoPimpleFoam."),
    PromptCase(prompt="transient compressible pipe Ma=0.35 D=0.04m L=0.4m rhoPimpleFoam", params=_compressible_pipe(0.35, 0.04, 0.4, is_transient=True), case_tag="rhoPimple_pipe_ma035", expert_notes="Ma=0.35 transient pipe rhoPimpleFoam."),
    PromptCase(prompt="rhoPimpleFoam pipe Ma=0.45 D=0.06m L=0.6m transient", params=_compressible_pipe(0.45, 0.06, 0.6, is_transient=True), case_tag="rhoPimple_pipe_ma045", expert_notes="Ma=0.45 transient pipe rhoPimpleFoam."),
    PromptCase(prompt="transient pipe Ma=0.5 D=0.05m L=0.5m air rhoPimpleFoam", params=_compressible_pipe(0.5, 0.05, 0.5, is_transient=True), case_tag="rhoPimple_pipe_ma05_med", expert_notes="Ma=0.5 transient pipe rhoPimpleFoam."),
    PromptCase(prompt="rhoPimpleFoam transient compressible pipe Ma=0.55 D=0.05m L=0.5m", params=_compressible_pipe(0.55, 0.05, 0.5, is_transient=True), case_tag="rhoPimple_pipe_ma055", expert_notes="Ma=0.55 transient pipe rhoPimpleFoam."),
    PromptCase(prompt="rhoPimpleFoam transient box Ma=0.3 wide 4m × 0.2m air", params=_compressible_box(0.3, 4.0, 0.2, is_transient=True), case_tag="rhoPimple_box_ma03_wide", expert_notes="Ma=0.3 wide transient box rhoPimpleFoam."),
    PromptCase(prompt="transient box Ma=0.4 narrow 1.5m × 0.05m rhoPimpleFoam air", params=_compressible_box(0.4, 1.5, 0.05, is_transient=True), case_tag="rhoPimple_box_ma04_narrow", expert_notes="Ma=0.4 narrow transient box rhoPimpleFoam."),
    PromptCase(prompt="rhoPimpleFoam transient pipe Ma=0.4 D=0.025m L=0.3m air", params=_compressible_pipe(0.4, 0.025, 0.3, is_transient=True), case_tag="rhoPimple_pipe_ma04_smallD", expert_notes="Ma=0.4 small-D transient pipe rhoPimpleFoam."),
    PromptCase(prompt="transient subsonic pipe Ma=0.35 D=0.06m L=0.7m rhoPimpleFoam", params=_compressible_pipe(0.35, 0.06, 0.7, is_transient=True), case_tag="rhoPimple_pipe_ma035_long", expert_notes="Ma=0.35 long transient pipe rhoPimpleFoam."),
    PromptCase(prompt="rhoPimpleFoam compressible box Ma=0.5 long 5m × 0.15m air", params=_compressible_box(0.5, 5.0, 0.15, is_transient=True), case_tag="rhoPimple_box_ma05_xl", expert_notes="Ma=0.5 XL transient box rhoPimpleFoam."),
    PromptCase(prompt="transient compressible box Ma=0.45 medium 2.5m × 0.12m rhoPimpleFoam", params=_compressible_box(0.45, 2.5, 0.12, is_transient=True), case_tag="rhoPimple_box_ma045_med", expert_notes="Ma=0.45 medium transient box rhoPimpleFoam."),

    # ── buoyantSimpleFoam augmented v2 ────────────────────────────────────
    PromptCase(prompt="natural convection cavity 0.6m × 0.6m hot left cold right air buoyantSimpleFoam", params=_buoyancy(0.6), case_tag="buoy_cavity_06m", expert_notes="0.6m square cavity laminar buoyancy."),
    PromptCase(prompt="buoyantSimpleFoam differentially heated cavity 0.8m × 0.8m air natural convection", params=_buoyancy(0.8), case_tag="buoy_cavity_08m", expert_notes="0.8m cavity laminar buoyancy."),
    PromptCase(prompt="natural convection 1.2m × 1.2m enclosure hot wall cold wall air buoyantSimpleFoam steady", params=_buoyancy(1.2), case_tag="buoy_cavity_12m", expert_notes="1.2m laminar cavity buoyantSimpleFoam."),
    PromptCase(prompt="buoyancy-driven flow 1.8m square air natural convection turbulent kOmegaSST buoyantSimpleFoam", params=_buoyancy(1.8, turb=True), case_tag="buoy_cavity_turb_18m", expert_notes="1.8m turbulent buoyancy kOmegaSST."),
    PromptCase(prompt="natural convection buoyantSimpleFoam 2.5m × 2.5m turbulent air high Rayleigh", params=_buoyancy(2.5, turb=True), case_tag="buoy_cavity_turb_25m", expert_notes="2.5m turbulent natural convection."),
    PromptCase(prompt="thermally driven cavity flow 0.4m × 0.4m air laminar buoyantSimpleFoam", params=_buoyancy(0.4), case_tag="buoy_cavity_04m", expert_notes="0.4m small cavity laminar."),
    PromptCase(prompt="buoyant air flow in 1m × 1m cavity steady-state heat transfer side walls 320K and 280K", params=_buoyancy(1.0), case_tag="buoy_cavity_1m_b", expert_notes="1m laminar with explicit T diff."),
    PromptCase(prompt="natural convection benchmark cavity 0.7m × 0.7m air buoyantSimpleFoam laminar", params=_buoyancy(0.7), case_tag="buoy_cavity_07m", expert_notes="0.7m laminar cavity buoyancy."),
    PromptCase(prompt="turbulent natural convection 1.5m × 1.5m differentially heated kOmegaSST buoyantSimpleFoam air", params=_buoyancy(1.5, turb=True), case_tag="buoy_cavity_turb_15m", expert_notes="1.5m turbulent buoyancy kOmegaSST."),
    PromptCase(prompt="indoor heated room CFD 3.5m × 2.5m air natural convection turbulent buoyantSimpleFoam", params=_buoyancy(3.5, turb=True, end_time=3000.0), case_tag="buoy_room_35x25", expert_notes="3.5m room turbulent buoyancy."),
    PromptCase(prompt="natural convection cavity 0.9m × 0.9m hot 360K cold 290K air buoyantSimpleFoam laminar", params=_buoyancy(0.9), case_tag="buoy_cavity_09m", expert_notes="0.9m laminar cavity 70K diff."),
    PromptCase(prompt="buoyantSimpleFoam steady 2.2m × 2.2m cavity turbulent natural convection air kOmegaSST", params=_buoyancy(2.2, turb=True), case_tag="buoy_cavity_turb_22m", expert_notes="2.2m turbulent buoyancy."),

    # ── interFoam augmented v2 ────────────────────────────────────────────
    PromptCase(prompt="dam break 1.5m × 1m water column collapse VOF interFoam multiphase", params=_dam_break(1.5, 1.0), case_tag="multiphase_dambreak_15x1", expert_notes="Dam break 1.5m × 1m. interFoam VOF."),
    PromptCase(prompt="dam break simulation 5m × 2.5m large domain water-air interFoam", params=_dam_break(5.0, 2.5), case_tag="multiphase_dambreak_5x25", expert_notes="Dam break 5m × 2.5m. interFoam."),
    PromptCase(prompt="interFoam dam break 6m × 3m water column collapse VOF", params=_dam_break(6.0, 3.0), case_tag="multiphase_dambreak_6x3", expert_notes="Dam break 6m × 3m. interFoam."),
    PromptCase(prompt="VOF dam break 2.5m × 1.5m water-air free surface flow interFoam", params=_dam_break(2.5, 1.5), case_tag="multiphase_dambreak_25x15", expert_notes="Dam break 2.5m × 1.5m. interFoam."),
    PromptCase(prompt="wave channel simulation 6m × 1.5m water-air VOF interFoam free surface", params=_wave_channel(6.0, 1.5), case_tag="multiphase_wave_6x15", expert_notes="Wave channel 6m × 1.5m. interFoam."),
    PromptCase(prompt="interFoam wave tank 10m × 2m water-air free surface tracking", params=_wave_channel(10.0, 2.0), case_tag="multiphase_wave_10x2", expert_notes="Wave tank 10m × 2m. interFoam."),
    PromptCase(prompt="free surface channel 12m × 2.5m water and air VOF interFoam", params=_wave_channel(12.0, 2.5), case_tag="multiphase_wave_12x25", expert_notes="Wave channel 12m × 2.5m. interFoam."),
    PromptCase(prompt="sloshing tank 1.5m × 0.75m partially filled water VOF interFoam transient", params=_sloshing(1.5, 0.75), case_tag="multiphase_slosh_15x075", expert_notes="Sloshing 1.5 × 0.75m. interFoam."),
    PromptCase(prompt="interFoam sloshing tank 2.5m × 1.25m water-air two-phase transient", params=_sloshing(2.5, 1.25), case_tag="multiphase_slosh_25x125", expert_notes="Sloshing 2.5 × 1.25m. interFoam."),
    PromptCase(prompt="liquid sloshing 3.5m × 1.75m container water and air VOF transient interFoam", params=_sloshing(3.5, 1.75), case_tag="multiphase_slosh_35x175", expert_notes="Sloshing 3.5 × 1.75m. interFoam."),
]
