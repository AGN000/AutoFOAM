from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schemas import CFDParams, TurbulenceModel, FlowRegime
from .numerical_policy import NumericalPolicy

_FOAM_HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2412                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       CLASS;
    object      OBJECT;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""

# Patch name → (U BC type, p BC type)
_PATCH_BC = {
    "inlet":        ("fixedValue", "zeroGradient"),
    "outlet":       ("inletOutlet", "fixedValue"),
    "walls":        ("noSlip", "zeroGradient"),
    "wall":         ("noSlip", "zeroGradient"),
    "cylinder":     ("noSlip", "zeroGradient"),
    "airfoil":      ("noSlip", "zeroGradient"),
    "step":         ("noSlip", "zeroGradient"),
    "movingwall":   ("fixedValue", "zeroGradient"),
    "fixedwalls":   ("noSlip", "zeroGradient"),
    "frontandback": ("empty", "empty"),
    "front":        ("wedge", "wedge"),
    "back":         ("wedge", "wedge"),
    "axis":         ("empty", "empty"),
    "symmetry":     ("symmetry", "symmetry"),
    "freestream":   ("freestreamVelocity", "freestreamPressure"),
    "farfield":     ("freestreamVelocity", "freestreamPressure"),
    "topwall":      ("noSlip", "zeroGradient"),
    "bottomwall":   ("noSlip", "zeroGradient"),
}


def _get_patch_bc(patch_name: str) -> tuple[str, str]:
    lower = patch_name.lower()
    if lower in _PATCH_BC:
        return _PATCH_BC[lower]
    # Fuzzy match
    for key, bc in _PATCH_BC.items():
        if key in lower:
            return bc
    if "wall" in lower:
        return ("noSlip", "zeroGradient")
    return ("zeroGradient", "zeroGradient")


def _header(cls: str, obj: str) -> str:
    return _FOAM_HEADER.replace("CLASS", cls).replace("OBJECT", obj)


def _read_patch_names(case_dir: Path) -> list[str]:
    boundary = case_dir / "constant" / "polyMesh" / "boundary"
    if not boundary.exists():
        return ["inlet", "outlet", "walls", "frontAndBack"]
    text = boundary.read_text()
    patches = []
    for line in text.splitlines():
        stripped = line.strip()
        if (stripped and not stripped.startswith("//")
                and not stripped.startswith("(")
                and stripped not in ("{", "}", "FoamFile")
                and "." not in stripped
                and stripped[0].isalpha()):
            words = stripped.split()
            if len(words) == 1:
                patches.append(words[0])
    return patches if patches else ["inlet", "outlet", "walls", "frontAndBack"]


@dataclass
class CaseWriterConfig:
    params: CFDParams
    solver: str
    case_dir: Path
    has_gmsh_mesh: bool = False
    nx: int = 40
    ny: int = 30
    nz: int = 1
    end_time: float = 1000
    delta_t: float = 1.0
    write_interval: int = 100
    numerical_policy: Optional[NumericalPolicy] = None


class CaseWriter:
    def write_all(self, cfg: CaseWriterConfig) -> list[Path]:
        p = cfg.params
        case_dir = cfg.case_dir
        for d in ("system", "constant", "0"):
            (case_dir / d).mkdir(parents=True, exist_ok=True)

        patches = _read_patch_names(case_dir) if cfg.has_gmsh_mesh else [
            "inlet", "outlet", "walls", "frontAndBack"
        ]

        files: dict[str, str] = {}
        files["system/controlDict"] = self._control_dict(cfg)
        files["system/fvSchemes"] = self._fv_schemes(cfg)
        files["system/fvSolution"] = self._fv_solution(cfg)
        if not cfg.has_gmsh_mesh:
            files["system/blockMeshDict"] = self._block_mesh_dict(cfg)
        if p.is_multiphase:
            files["constant/transportProperties"] = self._transport_props_multiphase(cfg)
        else:
            files["constant/transportProperties"] = self._transport_props(cfg)
        files["constant/turbulenceProperties"] = self._turbulence_props(cfg)
        files["0/U"] = self._u_field(cfg, patches)

        if p.is_multiphase:
            files["constant/g"] = self._g_field()
            files["0/p_rgh"] = self._p_rgh_field(cfg, patches)
            files["0/p"] = self._p_field(cfg, patches)
            files["0/alpha.water"] = self._alpha_field(cfg, patches)
        elif p.is_compressible:
            files["constant/thermophysicalProperties"] = self._thermo_props_compressible(cfg)
            files["0/p"] = self._p_field_compressible(cfg, patches)
            files["0/T"] = self._T_field_compressible(cfg, patches)
            files["0/alphat"] = self._alphat_field(cfg, patches)
        elif p.has_heat_transfer:
            files["0/p"] = self._p_field_buoyant(cfg, patches)
        else:
            files["0/p"] = self._p_field(cfg, patches)

        if p.turbulence_model in (TurbulenceModel.K_OMEGA_SST, TurbulenceModel.K_EPSILON):
            files.update(self._turbulence_fields(cfg, patches))
        if p.has_heat_transfer:
            files["constant/thermophysicalProperties"] = self._thermo_props(cfg)
            files["constant/g"] = self._g_field()
            files["0/p_rgh"] = self._p_rgh_field(cfg, patches)
            files["0/T"] = self._T_field(cfg, patches)
            files["0/alphat"] = self._alphat_field(cfg, patches)

        written = []
        for rel, content in files.items():
            fp = case_dir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content)
            written.append(fp)
        return written

    # ------------------------------------------------------------------ #
    #  system/controlDict
    # ------------------------------------------------------------------ #

    def _control_dict(self, cfg: CaseWriterConfig) -> str:
        p = cfg.params
        dt = cfg.delta_t if p.is_transient else 1
        end = cfg.end_time
        h = _header("dictionary", "controlDict")
        # interFoam and rhoPimpleFoam require adjustTimeStep to avoid CFL blowup
        extra = ""
        if p.is_multiphase:
            extra = """\
adjustTimeStep  yes;
maxCo           0.5;
maxAlphaCo      0.5;

"""
        elif p.is_compressible and p.is_transient:
            extra = """\
adjustTimeStep  yes;
maxCo           0.3;
maxDeltaT       1e-4;

"""
        return h + f"""\
application     {cfg.solver};

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end};

deltaT          {dt};
{extra}
writeControl    timeStep;
writeInterval   {cfg.write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    # ------------------------------------------------------------------ #
    #  system/fvSchemes
    # ------------------------------------------------------------------ #

    def _fv_schemes(self, cfg: CaseWriterConfig) -> str:
        p = cfg.params
        pol = cfg.numerical_policy
        is_turb = p.turbulence_model != TurbulenceModel.LAMINAR
        ddt = "Euler" if p.is_transient else "steadyState"
        if pol:
            div_u = pol.div_u
        else:
            # For transient laminar, linearUpwind is more robust on Gmsh unstructured meshes
            if is_turb:
                div_u = "Gauss limitedLinear 1"
            elif p.is_transient:
                div_u = "Gauss linearUpwind grad(U)"
            else:
                div_u = "Gauss linear"
        h = _header("dictionary", "fvSchemes")

        # Always required by OpenFOAM 2412 simpleFoam/pimpleFoam for viscous stress
        if p.is_multiphase:
            div_extra = "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n"
            div_extra += "    div(rhoPhi,U)           Gauss limitedLinear 1;\n"
            div_extra += "    div(rhoPhi,k)           Gauss limitedLinear 1;\n"
            div_extra += "    div(rhoPhi,omega)       Gauss limitedLinear 1;\n"
            div_extra += "    div(phi,alpha.water)    Gauss interfaceCompression vanLeer 1;\n"
            div_extra += "    div(phirb,alpha.water)  Gauss linear;\n"
            div_extra += "    div(phi,alpha)           Gauss interfaceCompression vanLeer 1;\n"
            div_extra += "    div(phirb,alpha)         Gauss linear;\n"
        elif p.is_compressible:
            # Pure upwind for compressible transient (rhoPimpleFoam) — more robust
            # than linearUpwind against CFL spikes in 2D wall-bounded ducts
            div_u = "bounded Gauss upwind" if p.is_transient else "bounded Gauss linearUpwind grad(U)"
            div_extra = "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n"
            # Using sensibleEnthalpy → div(phi,h); also need K and Ekp
            div_extra += "    div(phi,h)              bounded Gauss upwind;\n"
            div_extra += "    div(phi,K)              bounded Gauss upwind;\n"
            div_extra += "    div(phi,Ekp)            bounded Gauss upwind;\n"
            if p.is_transient:
                # rhoPimpleFoam requires div(phiv,p) for the pressure-velocity correction
                div_extra += "    div(phiv,p)             Gauss upwind;\n"
            else:
                # rhoSimpleFoam uses phid (density-weighted flux)
                div_extra += "    div(phid,p)             Gauss upwind;\n"
            div_extra += "    div((phi|interpolate(rho)),p) bounded Gauss upwind;\n"
        elif p.has_heat_transfer:
            div_u = "bounded Gauss upwind"
            div_extra = "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n"
            div_extra += "    div(phi,h)              bounded Gauss upwind;\n"
            div_extra += "    div(phi,K)              bounded Gauss upwind;\n"
            div_extra += "    div(phi,Ekp)            bounded Gauss upwind;\n"
        else:
            div_extra = "    div((nuEff*dev2(T(grad(U))))) Gauss linear;\n"
        if is_turb:
            if p.has_heat_transfer:
                div_k = "bounded Gauss upwind"
                div_oe = "bounded Gauss upwind"
            else:
                div_k = pol.div_k if pol else "Gauss limitedLinear 1"
                div_oe = pol.div_omega_eps if pol else "Gauss limitedLinear 1"
            div_extra += f"    div(phi,k)      {div_k};\n"
            if p.turbulence_model == TurbulenceModel.K_OMEGA_SST:
                div_extra += f"    div(phi,omega)  {div_oe};\n"
            elif p.turbulence_model == TurbulenceModel.K_EPSILON:
                div_extra += f"    div(phi,epsilon) {div_oe};\n"

        wall_dist = ""
        if is_turb:
            wall_dist = "\nwallDist\n{\n    method          meshWave;\n}\n"

        return h + f"""\
ddtSchemes
{{
    default         {ddt};
}}

gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,U)      {div_u};
    div(phi,p)      Gauss linear;
{div_extra}}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}{wall_dist}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    # ------------------------------------------------------------------ #
    #  system/fvSolution
    # ------------------------------------------------------------------ #

    def _fv_solution(self, cfg: CaseWriterConfig) -> str:
        p = cfg.params
        pol = cfg.numerical_policy
        solver = cfg.solver
        is_turb = p.turbulence_model != TurbulenceModel.LAMINAR
        h = _header("dictionary", "fvSolution")

        # Pull relaxation and corrector values from policy (with fallbacks)
        relax_p = pol.relax_p if pol else 0.4
        relax_U = pol.relax_U if pol else 0.7
        relax_k = pol.relax_k if pol else 0.5
        relax_oe = pol.relax_omega_eps if pol else 0.5
        n_corr = pol.n_correctors if pol else 2
        n_outer = pol.n_outer_correctors if pol else 2
        n_non_ortho = pol.n_non_ortho_correctors if pol else 2
        p_tol = pol.p_tolerance if pol else 1e-6
        u_tol = pol.U_tolerance if pol else 1e-5
        t_tol = pol.turb_tolerance if pol else 1e-5

        if p.has_heat_transfer or p.is_multiphase:
            p_field = "p_rgh"
        else:
            p_field = "p"

        extra_solvers = ""
        if p.is_multiphase:
            extra_solvers += f"""\
    "alpha.water.*"
    {{
        nAlphaCorr          2;
        nAlphaSubCycles     1;
        cAlpha              1;
    }}
    pcorr
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       {p_tol:.2e};
        relTol          0;
    }}
    pcorrFinal
    {{
        $pcorr;
        relTol          0;
    }}
"""
        if p.is_compressible:
            extra_solvers += f"""\
    rho
    {{
        solver          diagonal;
    }}
    rhoFinal
    {{
        $rho;
    }}
    h
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }}
    hFinal
    {{
        $h;
        relTol          0;
    }}
"""
        if p.has_heat_transfer and solver != "buoyantSimpleFoam":
            extra_solvers += f"""\
    h
    {{
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }}
    hFinal
    {{
        $h;
        relTol          0;
    }}
"""
        if solver == "rhoSimpleFoam":
            solvers = f"""\
solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.01;
    }}
    "(U|k|omega|epsilon|e)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-6;
        relTol          0.1;
    }}
{extra_solvers}}}
"""
        elif solver == "buoyantSimpleFoam":
            turb_regex = "k|epsilon|omega" if is_turb else "k"
            solvers = f"""\
solvers
{{
    p_rgh
    {{
        solver          GAMG;
        smoother        DICGaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }}
    "(U|h|{turb_regex})"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.1;
    }}
}}

"""
        else:
            solvers = f"""\
solvers
{{
    {p_field}
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       {p_tol:.2e};
        relTol          0;
    }}
    {p_field}Final
    {{
        ${p_field};
        relTol          0;
    }}
    U
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       {u_tol:.2e};
        relTol          0.1;
    }}
    UFinal
    {{
        $U;
        relTol          0;
    }}
{extra_solvers}"""
        _regex_solver = solver in ("rhoSimpleFoam", "buoyantSimpleFoam")
        if is_turb and not _regex_solver:
            solvers += f"""\
    k
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       {t_tol:.2e};
        relTol          0.1;
    }}
    kFinal
    {{
        $k;
        relTol          0;
    }}
"""
            if p.turbulence_model == TurbulenceModel.K_OMEGA_SST:
                solvers += f"""\
    omega
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       {t_tol:.2e};
        relTol          0.1;
    }}
    omegaFinal
    {{
        $omega;
        relTol          0;
    }}
"""
            elif p.turbulence_model == TurbulenceModel.K_EPSILON:
                solvers += f"""\
    epsilon
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       {t_tol:.2e};
        relTol          0.1;
    }}
    epsilonFinal
    {{
        $epsilon;
        relTol          0;
    }}
"""
        if not _regex_solver:
            solvers += "}\n\n"

        turb_relax = ""
        if is_turb:
            if p.turbulence_model == TurbulenceModel.K_OMEGA_SST:
                turb_relax = f"        k               {relax_k};\n        omega           {relax_oe};\n"
            else:
                turb_relax = f"        k               {relax_k};\n        epsilon         {relax_oe};\n"

        heat_relax = ""
        if solver == "buoyantSimpleFoam":
            heat_relax = "        h               0.2;\n"
        elif solver == "rhoSimpleFoam":
            heat_relax = "        e               0.7;\n"
        if solver == "rhoSimpleFoam":
            # Tutorial-matched setup: pMinFactor/pMaxFactor prevent T<0 divergence
            algo = f"""\
SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    pMinFactor      0.1;
    pMaxFactor      2;
}}

relaxationFactors
{{
    fields
    {{
        p               0.7;
        rho             0.01;
    }}
    equations
    {{
        U               0.3;
        e               0.7;
{turb_relax}    }}
}}
"""
        elif solver == "buoyantSimpleFoam":
            algo = f"""\
SIMPLE
{{
    nNonOrthogonalCorrectors {n_non_ortho};
    pRefCell        0;
    pRefValue       0;
}}

relaxationFactors
{{
    fields
    {{
        rho             1.0;
        p_rgh           0.7;
    }}
    equations
    {{
        U               0.2;
{heat_relax}{turb_relax}    }}
}}
"""
        elif solver == "simpleFoam":
            algo = f"""\
SIMPLE
{{
    nonOrthogonalCorrectors {n_non_ortho};
    pRefCell        0;
    pRefValue       0;
}}

relaxationFactors
{{
    fields
    {{
        {p_field}           {relax_p};
    }}
    equations
    {{
        U               {relax_U};
{heat_relax}{turb_relax}    }}
}}
"""
        elif solver in ("pimpleFoam", "buoyantPimpleFoam", "rhoPimpleFoam"):
            # For transient PIMPLE, lighter relaxation than SIMPLE is appropriate
            t_relax_p = min(1.0, max(0.7, relax_p + 0.3))
            t_relax_U = min(1.0, max(0.7, relax_U + 0.1))
            algo = f"""\
PIMPLE
{{
    nOuterCorrectors    {n_outer};
    nCorrectors         {n_corr};
    nNonOrthogonalCorrectors {n_non_ortho};
    pRefCell        0;
    pRefValue       0;
}}

relaxationFactors
{{
    fields {{ {p_field} {t_relax_p}; }}
    equations {{ U {t_relax_U}; }}
}}
"""
        elif solver == "icoFoam":
            algo = f"""\
PISO
{{
    nCorrectors     {n_corr};
    nNonOrthogonalCorrectors {n_non_ortho};
    pRefCell        0;
    pRefValue       0;
}}
"""
        elif solver == "interFoam":
            algo = """\
PIMPLE
{
    nCorrectors     3;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
"""
        else:
            algo = """\
SIMPLE
{
    nonOrthogonalCorrectors 2;
    pRefCell        0;
    pRefValue       0;
}
"""

        return h + solvers + algo + "\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"

    # ------------------------------------------------------------------ #
    #  system/blockMeshDict  (fallback when gmsh not used)
    # ------------------------------------------------------------------ #

    def _block_mesh_dict(self, cfg: CaseWriterConfig) -> str:
        p = cfg.params
        L, W = p.length, p.width
        H = 0.001 if not p.is_3d else p.height
        nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
        h = _header("dictionary", "blockMeshDict")
        return h + f"""\
convertToMeters 1;

vertices
(
    (0    0    0)
    ({L}  0    0)
    ({L}  {W}  0)
    (0    {W}  0)
    (0    0    {H})
    ({L}  0    {H})
    ({L}  {W}  {H})
    (0    {W}  {H})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 7 3));
    }}
    outlet
    {{
        type patch;
        faces ((1 2 6 5));
    }}
    walls
    {{
        type wall;
        faces ((0 1 5 4) (3 7 6 2));
    }}
    frontAndBack
    {{
        type empty;
        faces ((0 3 2 1) (4 5 6 7));
    }}
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    # ------------------------------------------------------------------ #
    #  constant/
    # ------------------------------------------------------------------ #

    def _transport_props(self, cfg: CaseWriterConfig) -> str:
        nu = cfg.params.kinematic_viscosity
        h = _header("dictionary", "transportProperties")
        return h + f"""\
transportModel  Newtonian;

nu              nu [0 2 -1 0 0 0 0] {nu};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _turbulence_props(self, cfg: CaseWriterConfig) -> str:
        p = cfg.params
        h = _header("dictionary", "turbulenceProperties")
        if p.turbulence_model == TurbulenceModel.LAMINAR:
            sim_type = "laminar"
            ras_block = ""
        elif p.turbulence_model == TurbulenceModel.K_OMEGA_SST:
            sim_type = "RAS"
            ras_block = "\nRAS\n{\n    RASModel        kOmegaSST;\n    turbulence      on;\n    printCoeffs     on;\n}\n"
        elif p.turbulence_model == TurbulenceModel.K_EPSILON:
            sim_type = "RAS"
            ras_block = "\nRAS\n{\n    RASModel        kEpsilon;\n    turbulence      on;\n    printCoeffs     on;\n}\n"
        elif p.turbulence_model == TurbulenceModel.LES_SMAGORINSKY:
            sim_type = "LES"
            ras_block = "\nLES\n{\n    LESModel        Smagorinsky;\n    delta           cubeRootVol;\n}\n"
        else:
            sim_type = "laminar"
            ras_block = ""

        return h + f"simulationType  {sim_type};\n{ras_block}\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"

    def _thermo_props(self, cfg: CaseWriterConfig) -> str:
        h = _header("dictionary", "thermophysicalProperties")
        return h + """\
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}

pRef            100000;

mixture
{
    specie          { molWeight 28.96; }
    thermodynamics  { Cp 1004.4; Hf 0; }
    transport       { mu 1.831e-05; Pr 0.705; }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _thermo_props_compressible(self, cfg: CaseWriterConfig) -> str:
        """thermophysicalProperties for rhoPimpleFoam / rhoSimpleFoam (perfect gas)."""
        h = _header("dictionary", "thermophysicalProperties")
        # sensibleEnthalpy is numerically more stable than sensibleInternalEnergy for
        # rhoPimpleFoam at higher Mach numbers (avoids negative T0 at first timestep)
        return h + """\
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}

mixture
{
    specie          { molWeight 28.9; }
    thermodynamics  { Cp 1005; Hf 0; }
    transport       { mu 1.8e-05; Pr 0.71; }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _transport_props_multiphase(self, cfg: CaseWriterConfig) -> str:
        """Two-phase transportProperties for interFoam."""
        h = _header("dictionary", "transportProperties")
        return h + """\
phases (water air);

water
{
    transportModel  Newtonian;
    nu              nu [0 2 -1 0 0 0 0] 1e-06;
    rho             rho [1 -3 0 0 0 0 0] 1000;
}

air
{
    transportModel  Newtonian;
    nu              nu [0 2 -1 0 0 0 0] 1.5e-05;
    rho             rho [1 -3 0 0 0 0 0] 1.225;
}

sigma           sigma [1 0 -2 0 0 0 0] 0.07;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _alpha_field(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        """0/alpha.water VOF field for interFoam."""
        h = _header("volScalarField", "alpha.water")
        bc_lines = []
        for patch in patches:
            lower = patch.lower()
            u_bc, _ = _get_patch_bc(patch)
            if u_bc == "empty":
                bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
            elif "outlet" in lower:
                bc_lines.append(f"    {patch}\n    {{\n        type            inletOutlet;\n        inletValue      uniform 0;\n        value           uniform 0;\n    }}\n")
            else:
                bc_lines.append(f"    {patch}\n    {{\n        type            zeroGradient;\n    }}\n")
        return h + f"""\
dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _alphat_field(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        """0/alphat — turbulent thermal diffusivity required by compressible solvers."""
        h = _header("volScalarField", "alphat")
        bc_lines = []
        for patch in patches:
            u_bc, _ = _get_patch_bc(patch)
            lower = patch.lower()
            if u_bc == "empty":
                bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
            elif u_bc == "wedge":
                bc_lines.append(f"    {patch}\n    {{\n        type            wedge;\n    }}\n")
            elif u_bc in ("symmetryPlane", "symmetry"):
                bc_lines.append(f"    {patch}\n    {{\n        type            symmetry;\n    }}\n")
            elif "wall" in lower or lower in ("cylinder", "airfoil", "step"):
                bc_lines.append(f"    {patch}\n    {{\n        type            compressible::alphatWallFunction;\n        Prt             0.85;\n        value           uniform 0;\n    }}\n")
            else:
                bc_lines.append(f"    {patch}\n    {{\n        type            calculated;\n        value           uniform 0;\n    }}\n")
        return h + f"""\
dimensions      [1 -1 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _p_field_compressible(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        """0/p in absolute pressure (Pa) for rhoPimpleFoam / rhoSimpleFoam."""
        h = _header("volScalarField", "p")
        # Non-reflective outlet BC for transient compressible (rhoPimpleFoam)
        # prevents acoustic wave reflection that accumulates continuity errors
        use_wave = cfg.params.is_transient
        bc_lines = []
        for patch in patches:
            u_bc, p_bc = _get_patch_bc(patch)
            if u_bc == "empty":
                bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
            elif u_bc == "wedge":
                bc_lines.append(f"    {patch}\n    {{\n        type            wedge;\n    }}\n")
            elif "outlet" in patch.lower():
                if use_wave:
                    bc_lines.append(
                        f"    {patch}\n    {{\n"
                        f"        type            waveTransmissive;\n"
                        f"        field           p;\n"
                        f"        phi             phi;\n"
                        f"        rho             rho;\n"
                        f"        psi             thermo:psi;\n"
                        f"        gamma           1.4;\n"
                        f"        fieldInf        101325;\n"
                        f"        lInf            {max(cfg.params.length * 2, 1.0):.4g};\n"
                        f"        value           uniform 101325;\n"
                        f"    }}\n"
                    )
                else:
                    bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform 101325;\n    }}\n")
            elif "inlet" in patch.lower():
                bc_lines.append(f"    {patch}\n    {{\n        type            zeroGradient;\n    }}\n")
            elif u_bc in ("symmetryPlane", "symmetry"):
                bc_lines.append(f"    {patch}\n    {{\n        type            symmetry;\n    }}\n")
            else:
                bc_lines.append(f"    {patch}\n    {{\n        type            zeroGradient;\n    }}\n")
        return h + f"""\
dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 101325;

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _p_field_buoyant(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        """0/p in absolute pressure (Pa) for buoyantSimpleFoam — all non-empty patches use 'calculated'."""
        h = _header("volScalarField", "p")
        bc_lines = []
        for patch in patches:
            u_bc, _ = _get_patch_bc(patch)
            if u_bc == "empty":
                bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
            elif u_bc in ("symmetryPlane", "symmetry"):
                bc_lines.append(f"    {patch}\n    {{\n        type            symmetry;\n    }}\n")
            else:
                bc_lines.append(f"    {patch}\n    {{\n        type            calculated;\n        value           $internalField;\n    }}\n")
        return h + f"""\
dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 1e5;

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _T_field_compressible(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        """0/T for rhoPimpleFoam — inlet fixed, outlet inletOutlet, walls zeroGradient."""
        h = _header("volScalarField", "T")
        bc_lines = []
        for patch in patches:
            u_bc, _ = _get_patch_bc(patch)
            lower = patch.lower()
            if u_bc == "empty":
                bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
            elif u_bc == "wedge":
                bc_lines.append(f"    {patch}\n    {{\n        type            wedge;\n    }}\n")
            elif u_bc in ("symmetryPlane", "symmetry"):
                bc_lines.append(f"    {patch}\n    {{\n        type            symmetry;\n    }}\n")
            elif "inlet" in lower:
                bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform 300;\n    }}\n")
            elif "outlet" in lower:
                bc_lines.append(f"    {patch}\n    {{\n        type            inletOutlet;\n        inletValue      uniform 300;\n        value           uniform 300;\n    }}\n")
            else:
                bc_lines.append(f"    {patch}\n    {{\n        type            zeroGradient;\n    }}\n")
        return h + f"""\
dimensions      [0 0 0 1 0 0 0];

internalField   uniform 300;

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _g_field(self) -> str:
        h = _header("uniformDimensionedVectorField", "g")
        return h + "dimensions      [0 1 -2 0 0 0 0];\n\nvalue           (0 -9.81 0);\n\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"

    # ------------------------------------------------------------------ #
    #  0/ fields — boundary condition generation
    # ------------------------------------------------------------------ #

    def _bc_block(self, patch: str, field: str, cfg: CaseWriterConfig) -> str:
        p = cfg.params
        U = p.inlet_velocity
        u_bc, p_bc = _get_patch_bc(patch)
        lower = patch.lower()

        if field == "U":
            if u_bc == "empty":
                return f"    {patch}\n    {{\n        type            empty;\n    }}\n"
            if u_bc == "wedge":
                return f"    {patch}\n    {{\n        type            wedge;\n    }}\n"
            if u_bc in ("symmetryPlane", "symmetry"):
                return f"    {patch}\n    {{\n        type            symmetry;\n    }}\n"
            if u_bc == "noSlip":
                return f"    {patch}\n    {{\n        type            noSlip;\n    }}\n"
            if u_bc == "fixedValue":
                if "moving" in lower:
                    return f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform ({U} 0 0);\n    }}\n"
                return f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform ({U} 0 0);\n    }}\n"
            if u_bc == "inletOutlet":
                return f"    {patch}\n    {{\n        type            inletOutlet;\n        inletValue      uniform (0 0 0);\n        value           uniform (0 0 0);\n    }}\n"
            if u_bc == "freestreamVelocity":
                return f"    {patch}\n    {{\n        type            freestreamVelocity;\n        freestreamValue uniform ({U} 0 0);\n    }}\n"
            return f"    {patch}\n    {{\n        type            zeroGradient;\n    }}\n"

        elif field == "p":
            if p_bc == "empty":
                return f"    {patch}\n    {{\n        type            empty;\n    }}\n"
            if p_bc == "wedge":
                return f"    {patch}\n    {{\n        type            wedge;\n    }}\n"
            if p_bc in ("symmetryPlane", "symmetry"):
                return f"    {patch}\n    {{\n        type            symmetry;\n    }}\n"
            if p_bc == "fixedValue":
                return f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform 0;\n    }}\n"
            if p_bc == "freestreamPressure":
                return (f"    {patch}\n    {{\n        type            freestreamPressure;\n"
                        f"        freestreamValue uniform 0;\n    }}\n")
            return f"    {patch}\n    {{\n        type            zeroGradient;\n    }}\n"
        return ""

    def _u_field(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        U = cfg.params.inlet_velocity
        h = _header("volVectorField", "U")
        if cfg.params.has_heat_transfer:
            bc_lines = []
            for patch in patches:
                u_bc, _ = _get_patch_bc(patch)
                if u_bc == "empty":
                    bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
                elif u_bc in ("symmetryPlane", "symmetry"):
                    bc_lines.append(f"    {patch}\n    {{\n        type            symmetry;\n    }}\n")
                else:
                    bc_lines.append(f"    {patch}\n    {{\n        type            noSlip;\n    }}\n")
            return h + f"""\
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""
        if cfg.params.is_multiphase:
            # For interFoam: inlet/walls are no-slip walls; outlet is atmosphere
            bc_lines = []
            for patch in patches:
                lower = patch.lower()
                u_bc, _ = _get_patch_bc(patch)
                if u_bc == "empty":
                    bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
                elif "outlet" in lower:
                    bc_lines.append(f"    {patch}\n    {{\n        type            pressureInletOutletVelocity;\n        value           uniform (0 0 0);\n    }}\n")
                else:
                    bc_lines.append(f"    {patch}\n    {{\n        type            noSlip;\n    }}\n")
            return h + f"""\
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""
        bc_text = "".join(self._bc_block(p, "U", cfg) for p in patches)
        # For compressible transient, start from rest — sudden high-Ma IC causes divergence
        u_init = "0 0 0" if (cfg.params.is_compressible and cfg.params.is_transient) else f"{U} 0 0"
        return h + f"""\
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform ({u_init});

boundaryField
{{
{bc_text}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _p_field(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        h = _header("volScalarField", "p")
        if cfg.params.is_multiphase:
            # For interFoam, p is derived from p_rgh; use 'calculated' everywhere
            bc_lines = []
            for patch in patches:
                u_bc, _ = _get_patch_bc(patch)
                if u_bc == "empty":
                    bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
                else:
                    bc_lines.append(f"    {patch}\n    {{\n        type            calculated;\n        value           $internalField;\n    }}\n")
            return h + f"""\
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""
        bc_text = "".join(self._bc_block(p, "p", cfg) for p in patches)
        return h + f"""\
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{bc_text}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _p_rgh_field(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        h = _header("volScalarField", "p_rgh")
        bc_lines = []
        is_multiphase = cfg.params.is_multiphase
        for patch in patches:
            lower = patch.lower()
            u_bc, _ = _get_patch_bc(patch)
            if u_bc == "empty":
                bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
            elif u_bc in ("symmetryPlane", "symmetry"):
                bc_lines.append(f"    {patch}\n    {{\n        type            symmetry;\n    }}\n")
            elif "outlet" in lower and is_multiphase:
                # Atmosphere patch for interFoam: totalPressure (gauge=0)
                bc_lines.append(f"    {patch}\n    {{\n        type            totalPressure;\n        p0              uniform 0;\n    }}\n")
            else:
                bc_lines.append(f"    {patch}\n    {{\n        type            fixedFluxPressure;\n        value           $internalField;\n    }}\n")
        # interFoam uses gauge pressure (p_rgh=0 at reference); buoyant solvers use absolute
        p_ref = "0" if is_multiphase else "1e5"
        return h + f"""\
dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform {p_ref};

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _T_field(self, cfg: CaseWriterConfig, patches: list[str]) -> str:
        h = _header("volScalarField", "T")
        bc_lines = []
        for patch in patches:
            lower = patch.lower()
            u_bc, _ = _get_patch_bc(patch)
            if u_bc == "empty":
                bc_lines.append(f"    {patch}\n    {{\n        type            empty;\n    }}\n")
            elif u_bc in ("symmetryPlane", "symmetry"):
                bc_lines.append(f"    {patch}\n    {{\n        type            symmetry;\n    }}\n")
            elif lower in ("lid", "movingwall"):
                bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform 350;\n    }}\n")
            elif "wall" in lower or lower in ("cylinder", "airfoil"):
                bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform 300;\n    }}\n")
            elif "inlet" in lower:
                bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform 300;\n    }}\n")
            else:
                bc_lines.append(f"    {patch}\n    {{\n        type            inletOutlet;\n        inletValue      uniform 300;\n        value           uniform 300;\n    }}\n")
        return h + f"""\
dimensions      [0 0 0 1 0 0 0];

internalField   uniform 300;

boundaryField
{{
{"".join(bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

    def _turbulence_fields(self, cfg: CaseWriterConfig, patches: list[str]) -> dict[str, str]:
        p = cfg.params
        U = p.inlet_velocity
        L = max(p.length, p.width) * 0.07
        turb_int = 0.05
        k_val = max(1e-6, 1.5 * (U * turb_int) ** 2)
        omg_bulk = max(1.0, k_val ** 0.5 / (0.09 ** 0.25 * L))
        eps_bulk = max(1e-6, k_val ** 1.5 / (0.09 ** 0.75 * L))
        fields = {}

        k_bc_lines = []
        omega_bc_lines = []
        eps_bc_lines = []
        nut_bc_lines = []

        for patch in patches:
            lower = patch.lower()
            u_bc, _ = _get_patch_bc(patch)

            if u_bc in ("empty", "wedge"):
                k_bc_lines.append(f"    {patch}\n    {{\n        type            {u_bc};\n    }}\n")
                omega_bc_lines.append(f"    {patch}\n    {{\n        type            {u_bc};\n    }}\n")
                eps_bc_lines.append(f"    {patch}\n    {{\n        type            {u_bc};\n    }}\n")
                nut_bc_lines.append(f"    {patch}\n    {{\n        type            {u_bc};\n    }}\n")
            elif u_bc in ("symmetryPlane", "symmetry"):
                for lst in (k_bc_lines, omega_bc_lines, eps_bc_lines, nut_bc_lines):
                    lst.append(f"    {patch}\n    {{\n        type            symmetry;\n    }}\n")
            elif "inlet" in lower or "freestream" in lower or "farfield" in lower:
                k_bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform {k_val};\n    }}\n")
                omega_bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform {omg_bulk};\n    }}\n")
                eps_bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform {eps_bulk};\n    }}\n")
                nut_bc_lines.append(f"    {patch}\n    {{\n        type            fixedValue;\n        value           uniform 0;\n    }}\n")
            elif "outlet" in lower:
                k_bc_lines.append(f"    {patch}\n    {{\n        type            inletOutlet;\n        inletValue      uniform {k_val};\n        value           uniform {k_val};\n    }}\n")
                omega_bc_lines.append(f"    {patch}\n    {{\n        type            inletOutlet;\n        inletValue      uniform {omg_bulk};\n        value           uniform {omg_bulk};\n    }}\n")
                eps_bc_lines.append(f"    {patch}\n    {{\n        type            inletOutlet;\n        inletValue      uniform {eps_bulk};\n        value           uniform {eps_bulk};\n    }}\n")
                nut_bc_lines.append(f"    {patch}\n    {{\n        type            zeroGradient;\n    }}\n")
            else:
                # Wall patches: proper wall functions prevent FP exceptions in kOmegaSST/kEpsilon
                k_bc_lines.append(f"    {patch}\n    {{\n        type            kqRWallFunction;\n        value           uniform {k_val};\n    }}\n")
                omega_bc_lines.append(f"    {patch}\n    {{\n        type            omegaWallFunction;\n        value           uniform {omg_bulk};\n    }}\n")
                eps_bc_lines.append(f"    {patch}\n    {{\n        type            epsilonWallFunction;\n        value           uniform {eps_bulk};\n    }}\n")
                nut_bc_lines.append(f"    {patch}\n    {{\n        type            nutkWallFunction;\n        value           uniform 0;\n    }}\n")

        # k field
        kh = _header("volScalarField", "k")
        fields["0/k"] = kh + f"""\
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_val};

boundaryField
{{
{"".join(k_bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

        if p.turbulence_model == TurbulenceModel.K_OMEGA_SST:
            omg_val = max(1.0, k_val ** 0.5 / (0.09 ** 0.25 * L))
            oh = _header("volScalarField", "omega")
            fields["0/omega"] = oh + f"""\
dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omg_val};

boundaryField
{{
{"".join(omega_bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""
        elif p.turbulence_model == TurbulenceModel.K_EPSILON:
            eps_val = max(1e-6, k_val ** 1.5 / (0.09 ** 0.75 * L))
            eh = _header("volScalarField", "epsilon")
            fields["0/epsilon"] = eh + f"""\
dimensions      [0 2 -3 0 0 0 0];

internalField   uniform {eps_val};

boundaryField
{{
{"".join(eps_bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

        nut_val = max(1e-6, 0.09 * k_val ** 2 / max(1e-10, k_val ** 1.5 / (0.09 ** 0.75 * L)))
        nth = _header("volScalarField", "nut")
        fields["0/nut"] = nth + f"""\
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform {nut_val};

boundaryField
{{
{"".join(nut_bc_lines)}}}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""
        return fields
