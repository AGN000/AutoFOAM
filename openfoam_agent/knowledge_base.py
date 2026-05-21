"""
Canonical OpenFOAM expert knowledge base.

Each entry is a structured document describing the physics, solver settings,
boundary conditions, and mesh strategy for one CFD case type.  These are
injected into ChromaDB alongside the tutorial cases so that RAG always
returns a relevant expert reference even when no tutorial exactly matches.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KnowledgeEntry:
    doc_id: str
    case_name: str       # shown to user and in RAG results
    content: str         # text indexed into ChromaDB


# ─────────────────────────────────────────────────────────────────────────────
# One entry per geometry type + key physics variant
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: list[KnowledgeEntry] = [

    # ── LID-DRIVEN CAVITY ────────────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_lid_driven_cavity",
        case_name="KnowledgeBase_LidDrivenCavity",
        content="""\
Case: 2D Lid-Driven Cavity Flow
Solver: simpleFoam (steady incompressible)
2D: True
Physics: lid-driven cavity laminar steady 2D benchmarck Ghia

Overview:
The lid-driven cavity (LDC) is the canonical benchmark for incompressible
Navier-Stokes solvers.  A square domain has a moving top lid and fixed walls.

Geometry: 1m × 1m square, 2D (height=0.001 for empty BC).
Typical Re range: 100–10000.  Laminar for Re<2300, turbulent above.

Solver settings (simpleFoam, steady):
- endTime: 2000 (pseudo-time steps)
- deltaT: 1
- SIMPLE algorithm, nNonOrthogonalCorrectors: 2

Boundary conditions (U):
  movingWall   fixedValue (0 1 0) × lid velocity
  fixedWalls   noSlip
  frontAndBack empty

Boundary conditions (p):
  movingWall   zeroGradient
  fixedWalls   zeroGradient
  frontAndBack empty

Mesh: blockMesh 50×50 uniform cells (Re=100–1000).
Lid velocity: U_lid = Re × nu / L

Numerical schemes (laminar):
  divSchemes: Gauss linear (Re<500) or Gauss limitedLinear 1 (Re>500)
  laplacianSchemes: Gauss linear corrected
  Relaxation: p=0.4, U=0.7

Validation: Compare U/U_lid along vertical centreline with Ghia 1982 data.
""",
    ),

    # ── TURBULENT PIPE FLOW ──────────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_pipe_flow",
        case_name="KnowledgeBase_PipeFlow",
        content="""\
Case: 3D Turbulent Pipe Flow
Solver: simpleFoam (steady incompressible turbulent)
2D: False
Physics: pipe flow turbulent laminar 3D kOmegaSST Hagen-Poiseuille circular duct

Overview:
Internal flow in a circular cross-section pipe.  Used to validate turbulence
models (kOmegaSST) and pressure-drop predictions (Moody chart).

Geometry: Circular pipe, D=0.05m, L=0.5m (typical).
Laminar Re < 2300 (Hagen-Poiseuille profile), turbulent Re > 4000.

Solver (simpleFoam, steady):
- endTime: 1000, deltaT: 1
- Turbulence: kOmegaSST with wall functions

Boundary conditions (U):
  inlet   fixedValue (U_inlet 0 0)
  outlet  zeroGradient
  walls   noSlip

Boundary conditions (p):
  inlet   zeroGradient
  outlet  fixedValue 0

Turbulence BCs:
  walls: kqRWallFunction (k), omegaWallFunction (omega), nutkWallFunction (nut)
  inlet: turbulentIntensityKineticEnergyInlet / turbulentMixingLengthFrequencyInlet

Mesh: gmsh 3D structured-like mesh, ~28k cells for D=0.05m.
Wall sizing: Distance/Threshold field, size_near_wall=D/10, size_bulk=D/3.
y+ target: 30 for wall functions.

Inlet velocity from Re: U = Re × nu / D
For D=0.05m air: U(Re=50000) = 50000 × 1.5e-5 / 0.05 = 15 m/s

Numerical schemes (turbulent):
  divSchemes: Gauss limitedLinear 1
  Relaxation: p=0.3, U=0.6, k=0.4, omega=0.4
""",
    ),

    # ── FLOW OVER CYLINDER ───────────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_cylinder_flow",
        case_name="KnowledgeBase_CylinderFlow",
        content="""\
Case: 2D Flow Over Circular Cylinder
Solver: simpleFoam (steady) or pimpleFoam (transient vortex shedding)
2D: True
Physics: cylinder cross-flow bluff body laminar wake vortex shedding drag lift 2D

Overview:
External flow over a circular cylinder.  Laminar steady Re<200; vortex
shedding (transient) Re≈100–400; turbulent Re>1000.

Geometry: D=0.1m cylinder, domain 20D long × 8D wide, 2D (height=0.001).
Reynolds number: Re = U × D / nu.

Solver:
  simpleFoam (steady, Re<400): endTime=1000, deltaT=1
  pimpleFoam (transient, Re≥100): deltaT=0.001, endTime=10

Boundary conditions (U):
  inlet      fixedValue (U 0 0)
  outlet     zeroGradient
  cylinder   noSlip
  freestream freestreamVelocity
  frontAndBack empty

Mesh: gmsh 2D with refinement ring around cylinder (r=2D, size=D/20 near wall).
Patch: cylinder (noSlip wall)

Forces post-processing: Drag Cd, Lift Cl via forces function object.
Benchmark Cd (Re=200): ~1.3–1.5

Numerical schemes (laminar, Re<500):
  divSchemes: Gauss linear
  Relaxation: p=0.4, U=0.7
""",
    ),

    # ── CHANNEL FLOW ─────────────────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_channel_flow",
        case_name="KnowledgeBase_ChannelFlow",
        content="""\
Case: 2D Plane Channel Flow (Poiseuille / turbulent)
Solver: simpleFoam (steady incompressible)
2D: True
Physics: channel flow plane Poiseuille turbulent laminar 2D rectangular duct

Overview:
Flow between two parallel plates.  Laminar Poiseuille (Re<2300) or turbulent
(Re>4000).  Re based on half-channel height h = H/2.

Geometry: L=5–10m, H=0.1m, 2D.
Typical Re: 1000 (laminar) to 50000 (turbulent).

Solver (simpleFoam, steady):
- endTime: 1000–2000
- Turbulence: kOmegaSST (turbulent) or laminar

Boundary conditions (U):
  inlet      fixedValue (U_inlet 0 0)
  outlet     zeroGradient
  walls      noSlip   (top and bottom walls — kqRWallFunction if turbulent)
  frontAndBack empty

U_inlet from Re: U = Re × nu / (H/2)   [Re based on half-height]

Laminar profile: u_max = 1.5 × U_mean (parabolic)
Turbulent log-law: u+ = (1/κ)·ln(y+) + B

Mesh: blockMesh uniform (50×20 laminar, 100×40 turbulent).
Turbulent wall BCs: kqRWallFunction, omegaWallFunction, nutkWallFunction.
""",
    ),

    # ── BACKWARD-FACING STEP ─────────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_bfs",
        case_name="KnowledgeBase_BackwardFacingStep",
        content="""\
Case: 2D Backward-Facing Step (BFS)
Solver: simpleFoam (steady incompressible)
2D: True
Physics: backward facing step separation reattachment recirculation Armaly benchmark

Overview:
Flow over an abrupt step expansion.  Separated shear layer reattaches
downstream.  Classic benchmark (Armaly 1983): Re=800, h=0.1m.

Geometry: step height h=0.1m, upstream length=0.2m, downstream=2m, 2D.
Expansion ratio: 2:1 (typical).  Re based on step height h.

Solver (simpleFoam, steady):
- endTime: 2000, deltaT: 1
- Laminar Re<2300; kOmegaSST Re>4000

Boundary conditions (U):
  inlet     fixedValue (U_inlet 0 0) — parabolic profile preferred
  outlet    zeroGradient
  step      noSlip
  topWall   noSlip
  bottomWall noSlip
  frontAndBack empty

Reattachment length: x_r/h ≈ 6 (Re=800, laminar Armaly)

Mesh: gmsh 2D BFS geometry.  Refine near step corner and bottom wall.
Step patch: noSlip (kqRWallFunction if turbulent).

Numerical schemes:
  divSchemes: Gauss linearUpwind (Re>500) for better accuracy near separation.
  Relaxation: p=0.4, U=0.7
""",
    ),

    # ── AIRFOIL FLOW ─────────────────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_airfoil",
        case_name="KnowledgeBase_AirfoilFlow",
        content="""\
Case: 2D Airfoil (NACA0012) External Flow
Solver: simpleFoam (steady RANS)
2D: True
Physics: airfoil NACA0012 lift drag angle of attack incidence external flow aerodynamics

Overview:
RANS simulation of 2D airfoil in external flow.  Used to compute lift (Cl)
and drag (Cd) coefficients.  Typical Re=1e5–1e6, AoA=0–15°.

Geometry: chord c=1m, far-field box 20c × 20c.
Inlet at x=-10c, outlet at x=+20c, walls at y=±10c.

Solver (simpleFoam, steady):
- endTime: 2000, deltaT: 1
- Turbulence: kOmegaSST (Re>1e5); laminar (Re<1e5)

Boundary conditions (U):
  freestream  freestreamVelocity (U_inf cos(AoA), U_inf sin(AoA), 0)
  outlet      zeroGradient
  airfoil     noSlip
  frontAndBack empty

Boundary conditions (p):
  freestream  freestreamPressure
  outlet      fixedValue 0
  airfoil     zeroGradient

Turbulence BCs (kOmegaSST):
  airfoil: kqRWallFunction (k), omegaWallFunction (omega), nutkWallFunction (nut)
  freestream: freestream (k=3/2(I·U)^2, omega=C_mu^0.5*k/nu_t)

Mesh: gmsh C-mesh around NACA0012. Refinement near leading edge and surface.
y+ target: 1 (wall-resolved) or 30 (wall functions).

Post-processing: forceCoeffs for Cl, Cd, Cm.
Stall: typically AoA > 14° for NACA0012 at Re=1e6.

Numerical schemes:
  divSchemes: Gauss linearUpwind grad(U)
  Relaxation: p=0.3, U=0.5 (aggressive — needed for AoA>5°)
""",
    ),

    # ── WEDGE (AXISYMMETRIC PIPE) ─────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_wedge",
        case_name="KnowledgeBase_WedgeAxisymmetric",
        content="""\
Case: Axisymmetric Pipe Flow (Wedge Boundary Conditions)
Solver: simpleFoam (steady incompressible)
2D: False (3D wedge)
Physics: axisymmetric wedge pipe flow axial symmetry Hagen-Poiseuille 5-degree wedge

Overview:
Axisymmetric simulation using OpenFOAM wedge BCs.  Equivalent to 3D pipe
but computationally cheaper — only a 5° wedge sector is simulated.

Geometry: 5° wedge, radius R=D/2, length L.  Revolves around x-axis.

Solver (simpleFoam, steady):
- endTime: 1000, deltaT: 1
- Turbulence: laminar (Re<2300) or kOmegaSST (Re>4000)

Boundary conditions (U):
  inlet  fixedValue (U_axial 0 0)
  outlet zeroGradient
  wall   noSlip
  axis   empty (axis of symmetry)
  front  wedge
  back   wedge

Boundary conditions (p):
  inlet  zeroGradient
  outlet fixedValue 0
  wall   zeroGradient
  axis   empty
  front  wedge
  back   wedge

CRITICAL: Both front and back must use 'type wedge' exactly.
axis patch must use 'type empty' and lie exactly on x-axis.

Mesh: gmsh 5° wedge via revolution.  Structured radial cells.
Typical R=0.025m, L=0.5m for D=0.05m pipe.

Validation: Laminar case → parabolic u(r) = 2·U_avg·(1-(r/R)^2).
""",
    ),

    # ── TURBULENT FLAT PLATE / BOX ────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_flat_plate",
        case_name="KnowledgeBase_FlatPlateBox",
        content="""\
Case: Flow Over Flat Plate / General Box Domain
Solver: simpleFoam (steady incompressible)
2D: True (or 3D)
Physics: flat plate box duct internal external boundary layer Blasius

Overview:
Simple rectangular domain used for flat-plate boundary layer development
or general duct/channel approximations.

Geometry: L × H box (or L × W × H for 3D).
Re based on plate length L: Re_L = U_inf × L / nu.

Solver (simpleFoam, steady):
- endTime: 1000–2000
- Turbulence: laminar (Re<2300) or kOmegaSST (Re>4000)

Boundary conditions (U):
  inlet      fixedValue (U_inlet 0 0)
  outlet     zeroGradient
  walls      noSlip (bottom plate) or symmetry (top/side far-field)
  frontAndBack empty (2D)

Mesh: blockMesh uniform (2D) or gmsh (3D).
Boundary layer grows from leading edge: δ ≈ 5x/sqrt(Re_x).

For Blasius laminar: Re_L < 5e5.
For turbulent: Re_L > 5e5, use kOmegaSST with wall functions.
""",
    ),

    # ── BUOYANCY-DRIVEN CAVITY ────────────────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_buoyancy_cavity",
        case_name="KnowledgeBase_BuoyancyCavity",
        content="""\
Case: Differentially Heated (Buoyancy-Driven) Cavity
Solver: buoyantSimpleFoam (steady) or buoyantPimpleFoam (transient)
2D: True
Physics: buoyancy natural convection differentially heated cavity temperature hot cold Rayleigh

Overview:
Natural convection in a square cavity with hot and cold vertical walls.
Governed by Rayleigh number Ra = g·β·ΔT·L³/(ν·α).

Geometry: 1m × 1m square, 2D.
Typical: hot wall T=350K, cold wall T=300K, ΔT=50K.

Solver (buoyantSimpleFoam, steady):
- endTime: 2000, deltaT: 1
- has_heat_transfer: True
- Requires: 0/T, 0/p_rgh, constant/thermophysicalProperties, constant/g

Boundary conditions:
  hotWall    fixedValue T=350K, noSlip U, zeroGradient p_rgh
  coldWall   fixedValue T=300K, noSlip U, zeroGradient p_rgh
  topWall    noSlip U, zeroGradient T, zeroGradient p_rgh
  bottomWall noSlip U, zeroGradient T, zeroGradient p_rgh
  frontAndBack empty

thermophysicalProperties: hePsiThermo, air, const transport model.
g: (0 -9.81 0)

Rayleigh number: Ra = g·β·ΔT·H³/(ν·α)
  For ΔT=50K, H=1m, air: Ra ≈ 1e10 (turbulent natural convection)

Validation: Nusselt number Nu vs Ra correlation (De Vahl Davis 1983).
""",
    ),

    # ── TRANSIENT CYLINDER VORTEX SHEDDING ────────────────────────────────────
    KnowledgeEntry(
        doc_id="kb_cylinder_transient",
        case_name="KnowledgeBase_VortexShedding",
        content="""\
Case: 2D Cylinder Vortex Shedding (Transient)
Solver: pimpleFoam (transient incompressible)
2D: True
Physics: vortex shedding cylinder transient unsteady von Karman street pimpleFoam Strouhal

Overview:
Transient simulation of laminar vortex shedding behind a circular cylinder.
Periodic shedding starts around Re=47; well-established at Re=100–300.

Geometry: D=0.1m cylinder, domain 20D × 8D, 2D.

Solver (pimpleFoam, transient):
- endTime: 10.0 seconds (several shedding cycles)
- deltaT: 0.001 (Courant < 1)
- nOuterCorrectors: 2, nCorrectors: 3

Boundary conditions — same as steady cylinder case.
Turbulence: laminar (Re<400).

Key physics:
  Strouhal number: St = f·D/U ≈ 0.2 (for Re=100–1000)
  Shedding frequency: f = St·U/D
  Lift Cl oscillates; Drag Cd shows small oscillations.

Mesh: same gmsh mesh as steady cylinder case — no adaptation needed.

Post-processing: forces function object to track Cl(t), Cd(t).
Monitor probeLocations: point at x=2D downstream for u(t) oscillation.
""",
    ),

]


def get_knowledge_entries() -> list[KnowledgeEntry]:
    return KNOWLEDGE_BASE
