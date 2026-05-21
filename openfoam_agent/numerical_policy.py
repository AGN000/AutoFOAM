from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import CFDParams, TurbulenceModel, FlowRegime


@dataclass
class NumericalPolicy:
    # Divergence schemes
    div_u: str = "Gauss limitedLinear 1"
    div_k: str = "Gauss limitedLinear 1"
    div_omega_eps: str = "Gauss limitedLinear 1"
    # Laplacian
    laplacian: str = "Gauss linear corrected"
    # SIMPLE relaxation
    relax_p: float = 0.4
    relax_U: float = 0.7
    relax_k: float = 0.5
    relax_omega_eps: float = 0.5
    # Transient correctors
    n_correctors: int = 2
    n_outer_correctors: int = 2
    n_non_ortho_correctors: int = 2
    # Boundary layer (used by gmsh generator)
    y_plus_target: float = 1.0
    first_cell_height: float = 1e-4
    bl_layers: int = 5
    bl_growth_rate: float = 1.3
    # Solver tolerances
    p_tolerance: float = 1e-6
    U_tolerance: float = 1e-5
    turb_tolerance: float = 1e-5


def compute_numerical_policy(params: CFDParams, solver: str) -> NumericalPolicy:
    pol = NumericalPolicy()
    re = max(params.reynolds_number or 1000.0, 1.0)
    nu = params.kinematic_viscosity
    U = params.inlet_velocity
    char_len = params.diameter or params.width

    # --- Divergence schemes: stability vs. accuracy tradeoff ---
    if re < 500:
        pol.div_u = "Gauss linear"
        pol.div_k = "Gauss linear"
        pol.div_omega_eps = "Gauss linear"
    elif re < 5_000:
        pol.div_u = "Gauss linearUpwind grad(U)"
        pol.div_k = "Gauss limitedLinear 1"
        pol.div_omega_eps = "Gauss limitedLinear 1"
    elif re < 100_000:
        pol.div_u = "Gauss limitedLinear 1"
        pol.div_k = "Gauss upwind"
        pol.div_omega_eps = "Gauss upwind"
    else:
        pol.div_u = "Gauss upwind"
        pol.div_k = "Gauss upwind"
        pol.div_omega_eps = "Gauss upwind"

    # --- Relaxation: tighter for higher Re (harder to converge) ---
    if re < 1_000:
        pol.relax_p, pol.relax_U = 0.6, 0.8
        pol.relax_k = pol.relax_omega_eps = 0.7
    elif re < 10_000:
        pol.relax_p, pol.relax_U = 0.4, 0.7
        pol.relax_k = pol.relax_omega_eps = 0.5
    elif re < 100_000:
        pol.relax_p, pol.relax_U = 0.3, 0.6
        pol.relax_k = pol.relax_omega_eps = 0.4
    else:
        pol.relax_p, pol.relax_U = 0.2, 0.5
        pol.relax_k = pol.relax_omega_eps = 0.3

    # --- y+ target (wall-normal first cell size) ---
    if params.turbulence_model == TurbulenceModel.K_EPSILON:
        pol.y_plus_target = 30.0  # wall-function regime
    else:
        pol.y_plus_target = 1.0  # wall-resolved (kOmegaSST / laminar)

    # Estimate friction velocity: u_tau = U * sqrt(Cf/2)
    if re > 4_000:
        cf = 0.026 * re ** (-1.0 / 7.0)  # turbulent pipe/flat-plate estimate
    else:
        cf = 0.664 / math.sqrt(re)  # Blasius laminar
    u_tau = U * math.sqrt(max(cf / 2.0, 1e-12))

    if u_tau > 1e-12 and nu > 0:
        pol.first_cell_height = pol.y_plus_target * nu / u_tau
        pol.first_cell_height = float(
            min(max(pol.first_cell_height, 1e-8), char_len / 20.0)
        )
    else:
        pol.first_cell_height = char_len / 100.0

    # BL layers: grow until covering ~15% of char_len
    bl_target = 0.15 * char_len
    gr = 1.3
    pol.bl_growth_rate = gr
    y1 = pol.first_cell_height
    if 0 < y1 < bl_target:
        n = math.log(bl_target / y1 * (gr - 1) + 1) / math.log(gr)
        pol.bl_layers = min(max(int(n), 3), 25)
    else:
        pol.bl_layers = 5

    # --- PIMPLE / PISO correctors ---
    if solver in ("pimpleFoam", "rhoPimpleFoam", "buoyantPimpleFoam"):
        pol.n_correctors = 3 if re > 50_000 else 2
        pol.n_outer_correctors = 3 if re > 100_000 else 2
    elif solver == "icoFoam":
        pol.n_correctors = 4
        pol.n_outer_correctors = 1
    # Non-orthogonal correctors — keep at 2 for safety
    pol.n_non_ortho_correctors = 2

    # --- Solver tolerances ---
    if re < 500:
        pol.p_tolerance = 1e-8
        pol.U_tolerance = 1e-7
    else:
        pol.p_tolerance = 1e-6
        pol.U_tolerance = 1e-5
    pol.turb_tolerance = 1e-5

    return pol
