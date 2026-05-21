from __future__ import annotations

from .schemas import CFDParams, FlowRegime


def select_solver(params: CFDParams) -> str:
    if params.is_multiphase:
        return "interFoam"
    if params.is_compressible:
        return "rhoPimpleFoam" if params.is_transient else "rhoSimpleFoam"
    if params.has_heat_transfer:
        return "buoyantPimpleFoam" if params.is_transient else "buoyantSimpleFoam"
    if params.is_transient:
        if (params.flow_regime == FlowRegime.LAMINAR
                and params.reynolds_number is not None
                and params.reynolds_number < 2300):
            return "icoFoam"
        return "pimpleFoam"
    return "simpleFoam"


def compute_mesh_resolution(params: CFDParams) -> dict[str, int]:
    L, W, H = params.length, params.width, params.height
    re = params.reynolds_number or 1000.0

    base = max(20, min(80, int(re ** 0.25 * 5)))
    aspect = L / W if W > 0 else 1.0

    if params.is_3d:
        # Cap 3D mesh to ~100k cells to keep blockMesh fallback tractable
        nx = min(60, max(10, int(base * min(aspect, 3))))
        ny = min(40, base)
        nz = max(5, base // 4)
    else:
        nx = min(200, max(20, int(base * aspect)))
        ny = base
        nz = 1

    return {"nx": nx, "ny": ny, "nz": nz}


def compute_time_settings(params: CFDParams, solver: str) -> dict:
    from .schemas import GeometryType
    if not params.is_transient or solver in ("simpleFoam", "rhoSimpleFoam",
                                              "buoyantSimpleFoam"):
        return {"end_time": 1000, "delta_t": 1, "write_interval": 100}

    U = max(params.inlet_velocity, 1e-6)
    res = compute_mesh_resolution(params)
    dx_bulk = params.length / res["nx"]

    # For cylinder/bluff-body geometries the gmsh near-wall mesh is much finer
    # than the bulk dx — use the cylinder diameter as the characteristic cell size
    if params.geometry_type == GeometryType.CYLINDER:
        D = params.diameter or params.width
        # For laminar icoFoam/pimpleFoam: strict CFL<1 → D/20
        # For turbulent pimpleFoam with kOmegaSST: CFL<2 is OK → D/10
        from .schemas import FlowRegime
        if params.flow_regime == FlowRegime.LAMINAR:
            dx = D / 20.0
        else:
            dx = D / 10.0
    else:
        dx = dx_bulk

    cfl_dt = 0.4 * dx / U
    delta_t = max(1e-5, min(0.1, cfl_dt))
    end_time = params.end_time if params.end_time < 1e6 else 10.0
    write_interval = max(1, int(end_time / delta_t / 20))

    return {"end_time": end_time, "delta_t": round(delta_t, 6),
            "write_interval": write_interval}
