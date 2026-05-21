from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from .schemas import CFDParams, GeometryType
from .config import OPENFOAM_BASHRC
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .numerical_policy import NumericalPolicy

_gmsh_lock = threading.Lock()  # gmsh is not thread-safe


def _run_gmsh_in_subprocess(params_json: str, case_dir: str, geometry_type: str) -> None:
    """Entry point called in a fresh subprocess to isolate gmsh from vllm's process space."""
    import json
    from pathlib import Path

    # Re-import locally so the parent process never loads gmsh
    data = json.loads(params_json)

    # Reconstruct params
    from openfoam_agent.schemas import CFDParams
    params = CFDParams.model_validate(data)

    gen = GmshMeshGenerator(_subprocess_mode=True)
    gen.generate(params, Path(case_dir))

PATCH_TYPE_MAP = {
    "frontandback": "empty",
    "front": "wedge",
    "back": "wedge",
    "axis": "empty",
    "symmetry": "symmetry",
}


def _patch_type(name: str) -> str:
    lower = name.lower()
    for key, ptype in PATCH_TYPE_MAP.items():
        if lower == key or lower.startswith(key):
            return ptype
    if "wall" in lower or lower in ("cylinder", "airfoil", "step", "sphere"):
        return "wall"
    return "patch"


class GmshMeshGenerator:
    def __init__(self, _subprocess_mode: bool = False):
        self._subprocess_mode = _subprocess_mode

    def generate(self, params: CFDParams, case_dir: Path, policy=None) -> Path:
        case_dir.mkdir(parents=True, exist_ok=True)

        # If vllm is already loaded in this process, gmsh's nanobind will conflict
        # with vllm's forked engine core. Run gmsh in a clean subprocess instead.
        if not self._subprocess_mode and "vllm" in sys.modules:
            return self._generate_in_subprocess(params, case_dir, policy)

        msh_file = case_dir / "mesh.msh"
        self._policy = policy  # stored for use in builders

        builder = {
            GeometryType.BOX: self._build_box,
            GeometryType.CHANNEL: self._build_box,
            GeometryType.LID_DRIVEN_CAVITY: self._build_lid_cavity,
            GeometryType.CYLINDER: self._build_cylinder_2d,
            GeometryType.PIPE: self._build_pipe_3d,
            GeometryType.BACKWARD_FACING_STEP: self._build_bfs,
            GeometryType.AIRFOIL: self._build_airfoil_box,
            GeometryType.WEDGE: self._build_wedge,
            GeometryType.PERIODIC_HILL: self._build_pehill,
            GeometryType.S_BEND: self._build_sbend,
            GeometryType.DIFFUSER: self._build_diffuser,
            GeometryType.SPHERE: self._build_sphere_3d,
            GeometryType.AHMED_BODY: self._build_ahmed_body,
            GeometryType.MULTI_HILL: self._build_multi_hill,
            GeometryType.T_JUNCTION: self._build_tjunction,
            GeometryType.CD_NOZZLE: self._build_cd_nozzle,
            GeometryType.ELBOW: self._build_elbow,
        }.get(params.geometry_type, self._build_box)

        with _gmsh_lock:
            builder(params, msh_file)

        self._run_gmsh_to_foam(msh_file, case_dir)
        self._fix_boundary_file(case_dir)
        msh_file.unlink(missing_ok=True)
        return case_dir / "constant" / "polyMesh"

    def _generate_in_subprocess(self, params: CFDParams, case_dir: Path, policy=None) -> Path:
        """Spawn a fresh Python process to run gmsh, avoiding nanobind/vllm conflict."""
        params_json = params.model_dump_json()
        # Serialize policy as simple dict if provided
        policy_json = "None"
        if policy is not None:
            import dataclasses
            policy_json = repr(dataclasses.asdict(policy))
        script = (
            "import sys, json; "
            "sys.path.insert(0, '/data/foamllm3/openfoam_agent'); "
            "from openfoam_agent.gmsh_generator import GmshMeshGenerator; "
            "from openfoam_agent.schemas import CFDParams; "
            "from openfoam_agent.numerical_policy import NumericalPolicy; "
            "from pathlib import Path; "
            f"params = CFDParams.model_validate_json({repr(params_json)}); "
            f"pol_dict = {policy_json}; "
            "pol = NumericalPolicy(**pol_dict) if pol_dict else None; "
            f"gen = GmshMeshGenerator(_subprocess_mode=True); "
            f"gen.generate(params, Path({repr(str(case_dir))}), pol)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gmsh subprocess failed:\n{result.stderr[-2000:]}")
        return case_dir / "constant" / "polyMesh"

    # ------------------------------------------------------------------ #
    #  Shared mesh-sizing helper (the "good procedure" from PROJECT_STATUS)
    # ------------------------------------------------------------------ #

    def _mesh_sizing(self, params: CFDParams, char_len: float) -> tuple[float, float]:
        """Return (size_near_wall, size_bulk) following the Distance/Threshold
        procedure documented in PROJECT_STATUS.md:
          - Use NumericalPolicy first_cell_height if available
          - Clamp near-wall size to char_len/10 min to prevent 100k+ cell counts
          - size_bulk = char_len / 3
        """
        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else char_len / 10
        size_near_wall = max(size_near_wall, char_len / 10)   # never finer than L/10
        size_bulk = max(char_len / 3, size_near_wall * 4)     # bulk always coarser
        return size_near_wall, size_bulk

    def _apply_wall_field(self, wall_tags: list, char_len: float, dist_max_frac: float = 0.3):
        """Attach a Distance/Threshold background mesh field to wall_tags."""
        import gmsh
        if not wall_tags:
            return
        size_near_wall, size_bulk = self._mesh_sizing(
            getattr(self, "_current_params", None) or object(), char_len
        )
        dist_f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_f, "SurfacesList", wall_tags)
        th_f = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(th_f, "InField", dist_f)
        gmsh.model.mesh.field.setNumber(th_f, "SizeMin", size_near_wall)
        gmsh.model.mesh.field.setNumber(th_f, "SizeMax", size_bulk)
        gmsh.model.mesh.field.setNumber(th_f, "DistMin", 0)
        gmsh.model.mesh.field.setNumber(th_f, "DistMax", char_len * dist_max_frac)
        gmsh.model.mesh.field.setAsBackgroundMesh(th_f)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

    # ------------------------------------------------------------------ #
    #  Geometry builders
    # ------------------------------------------------------------------ #

    def _build_box(self, params: CFDParams, msh_file: Path):
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("box")
        self._current_params = params

        L, W = params.length, params.width
        depth = 0.001 if not params.is_3d else params.height
        # char_len for sizing = cross-stream dimension (W for channel, H for 3D duct)
        char_len = W

        try:
            if params.is_3d:
                vol = gmsh.model.occ.addBox(0, 0, 0, L, W, depth)
                gmsh.model.occ.synchronize()
                surfs = gmsh.model.occ.getEntities(2)
                tol3d = max(1e-4 * L, 1e-6)
                inlet_tags, outlet_tags, wall_tags = [], [], []
                for dim, tag in surfs:
                    bb = gmsh.model.occ.getBoundingBox(dim, tag)
                    xmin, xmax = bb[0], bb[3]
                    if abs(xmin) < tol3d and abs(xmax) < tol3d:
                        inlet_tags.append(tag)
                    elif abs(xmin - L) < tol3d and abs(xmax - L) < tol3d:
                        outlet_tags.append(tag)
                    else:
                        wall_tags.append(tag)
                if inlet_tags:
                    gmsh.model.addPhysicalGroup(2, inlet_tags, name="inlet")
                if outlet_tags:
                    gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
                if wall_tags:
                    gmsh.model.addPhysicalGroup(2, wall_tags, name="walls")
                gmsh.model.addPhysicalGroup(3, [vol], name="fluid")
                # Wall refinement for 3D turbulent duct
                if wall_tags:
                    self._apply_wall_field(wall_tags, char_len, dist_max_frac=0.5)
                gmsh.model.mesh.generate(3)
            else:
                rect = gmsh.model.occ.addRectangle(0, 0, 0, L, W)
                gmsh.model.occ.synchronize()
                ext = gmsh.model.occ.extrude(
                    [(2, rect)], 0, 0, depth, numElements=[1], recombine=True
                )
                gmsh.model.occ.synchronize()
                vol_tag = ext[1][1]
                front_tag = ext[0][1]
                lat_surfs = [(d, t) for d, t in ext[2:] if d == 2]

                tol = max(1e-4 * L, 1e-6)  # Gmsh bbox adds ~1e-7 padding
                inlet_tags, outlet_tags, wall_tags = [], [], []
                for _, tag in lat_surfs:
                    bb = gmsh.model.occ.getBoundingBox(2, tag)
                    xmin, xmax = bb[0], bb[3]
                    if abs(xmax - xmin) < tol and abs(xmin) < tol:
                        inlet_tags.append(tag)
                    elif abs(xmax - xmin) < tol and abs(xmax - L) < tol:
                        outlet_tags.append(tag)
                    else:
                        wall_tags.append(tag)

                gmsh.model.addPhysicalGroup(2, [rect, front_tag], name="frontAndBack")
                if inlet_tags:
                    gmsh.model.addPhysicalGroup(2, inlet_tags, name="inlet")
                if outlet_tags:
                    gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
                if wall_tags:
                    gmsh.model.addPhysicalGroup(2, wall_tags, name="walls")
                gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")
                # 2D: add near-wall refinement when turbulent (Re > 4000)
                re = params.reynolds_number or 0
                if re > 4000 and wall_tags:
                    self._apply_wall_field(wall_tags, char_len, dist_max_frac=0.4)
                else:
                    # Uniform mesh size for laminar 2D box
                    lc = min(W / 20, L / 50)
                    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
                gmsh.model.mesh.generate(3)

            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_lid_cavity(self, params: CFDParams, msh_file: Path):
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("cavity")
        self._current_params = params

        L, W = params.length, params.width
        depth = 0.001
        # Re-aware mesh density: 40 cells at Re=100, 80 at Re=3200, cap at 100
        re = params.reynolds_number or 100
        n_cells = int(min(max(40 * (re / 100) ** 0.25, 40), 100))
        lc = min(L, W) / n_cells

        try:
            rect = gmsh.model.occ.addRectangle(0, 0, 0, L, W)
            gmsh.model.occ.synchronize()
            ext = gmsh.model.occ.extrude(
                [(2, rect)], 0, 0, depth, numElements=[1], recombine=True
            )
            gmsh.model.occ.synchronize()

            vol_tag = ext[1][1]
            front_tag = ext[0][1]
            lat_surfs = [(d, t) for d, t in ext[2:] if d == 2]

            moving_tags, fixed_tags = [], []
            pos_tol = max(L, W) * 1e-4  # tolerance for y-position classification
            zspan_tol = depth * 0.1     # tolerance to exclude the z=0 base face (zspan≈0)
            for _, tag in lat_surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                yc = (bb[1] + bb[4]) / 2
                zspan = abs(bb[5] - bb[2])
                if zspan < zspan_tol:
                    continue
                if abs(yc - W) < pos_tol:
                    moving_tags.append(tag)
                else:
                    fixed_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [rect, front_tag], name="frontAndBack")
            gmsh.model.addPhysicalGroup(2, moving_tags, name="movingWall")
            gmsh.model.addPhysicalGroup(2, fixed_tags, name="fixedWalls")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            # Near-wall refinement for higher Re using Distance/Threshold on walls
            all_wall_tags = moving_tags + fixed_tags
            if re > 500 and all_wall_tags:
                # Finer near the lid and fixed walls — separation scale ~ L/Re^0.5
                size_wall = max(lc * 0.4, L / 200)
                dist_f = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_f, "SurfacesList", all_wall_tags)
                th_f = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(th_f, "InField", dist_f)
                gmsh.model.mesh.field.setNumber(th_f, "SizeMin", size_wall)
                gmsh.model.mesh.field.setNumber(th_f, "SizeMax", lc)
                gmsh.model.mesh.field.setNumber(th_f, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(th_f, "DistMax", W * 0.15)
                gmsh.model.mesh.field.setAsBackgroundMesh(th_f)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
            else:
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_cylinder_2d(self, params: CFDParams, msh_file: Path):
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("cylinder2d")

        D = params.diameter or params.width
        R = D / 2.0
        depth = 0.001

        # Domain extents in cylinder diameters. These are the standard
        # incompressible-cylinder benchmark proportions (DNS / VIV literature):
        #   upstream   ≥ 10 D   to fully develop the freestream
        #   downstream ≥ 25 D   to capture the Kármán wake
        #   crossflow  ≥ 10 D   each side, so blockage stays ≤ 5 %
        # We honour user overrides only if they exceed these minimums.
        upstream   = max(10.0 * D, 10.0 * D)
        downstream = max(params.length, 25.0 * D)
        halfwidth  = max(params.width / 2.0, 10.0 * D)
        x0, x1 = -upstream, downstream
        y0, y1 = -halfwidth, halfwidth
        L = x1 - x0   # used by downstream meshing-density logic
        W = 2 * halfwidth

        try:
            rect = gmsh.model.occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)
            disk = gmsh.model.occ.addDisk(0, 0, 0, R, R)
            domain, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
            gmsh.model.occ.synchronize()

            surfs = gmsh.model.occ.getEntities(2)
            surf_tag = surfs[0][1] if surfs else 1

            # Extrude
            ext = gmsh.model.occ.extrude(
                [(2, surf_tag)], 0, 0, depth, numElements=[1], recombine=True
            )
            gmsh.model.occ.synchronize()

            vol_tag = ext[1][1]
            front_tag = ext[0][1]
            lat_surfs = [(d, t) for d, t in ext[2:] if d == 2]

            inlet_tags, outlet_tags, top_tags, bot_tags, cyl_tags = [], [], [], [], []
            for _, tag in lat_surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xmid = (bb[0] + bb[3]) / 2
                ymid = (bb[1] + bb[4]) / 2
                xspan = abs(bb[3] - bb[0])
                yspan = abs(bb[4] - bb[1])

                if xspan < 1e-6 and abs(bb[0] - x0) < 0.01:
                    inlet_tags.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - x1) < 0.01:
                    outlet_tags.append(tag)
                elif yspan < 1e-6 and bb[1] > 0:
                    top_tags.append(tag)
                elif yspan < 1e-6 and bb[4] < 0:
                    bot_tags.append(tag)
                else:
                    cyl_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf_tag, front_tag], name="frontAndBack")
            if inlet_tags:
                gmsh.model.addPhysicalGroup(2, inlet_tags, name="inlet")
            if outlet_tags:
                gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            sym_tags = top_tags + bot_tags
            if sym_tags:
                gmsh.model.addPhysicalGroup(2, sym_tags, name="symmetry")
            if cyl_tags:
                gmsh.model.addPhysicalGroup(2, cyl_tags, name="cylinder")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            # Physics-aware near-wall refinement
            pol = getattr(self, "_policy", None)
            size_min = pol.first_cell_height if pol else D / 20
            size_min = max(size_min, D / 50)  # avoid degenerate cells
            f = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(f, "SurfacesList", cyl_tags or [1])
            th = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(th, "InField", f)
            gmsh.model.mesh.field.setNumber(th, "SizeMin", size_min)
            gmsh.model.mesh.field.setNumber(th, "SizeMax", D * 2)
            gmsh.model.mesh.field.setNumber(th, "DistMin", R)
            gmsh.model.mesh.field.setNumber(th, "DistMax", 5 * D)
            gmsh.model.mesh.field.setAsBackgroundMesh(th)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_pipe_3d(self, params: CFDParams, msh_file: Path):
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("pipe3d")

        D = params.diameter or params.width
        R = D / 2.0
        L = params.length

        try:
            cyl = gmsh.model.occ.addCylinder(0, 0, 0, L, 0, 0, R)
            gmsh.model.occ.synchronize()

            surfs = gmsh.model.occ.getEntities(2)
            inlet_tags, outlet_tags, wall_tags = [], [], []
            for _, tag in surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0])
                if xspan < 1e-6 and bb[0] < 0.01:
                    inlet_tags.append(tag)
                elif xspan < 1e-6 and bb[0] > L - 0.01:
                    outlet_tags.append(tag)
                else:
                    wall_tags.append(tag)

            if inlet_tags:
                gmsh.model.addPhysicalGroup(2, inlet_tags, name="inlet")
            if outlet_tags:
                gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if wall_tags:
                gmsh.model.addPhysicalGroup(2, wall_tags, name="wall")
            gmsh.model.addPhysicalGroup(3, [cyl], name="fluid")

            # Physics-aware mesh sizing from numerical policy.
            # For unstructured 3D meshes, y+=1 is impractically fine.
            # Clamp to D/30 minimum to keep cell count manageable.
            pol = getattr(self, "_policy", None)
            size_near_wall = pol.first_cell_height if pol else D / 10
            # For unstructured 3D, y+-resolved sizing creates 100k+ cells → timeout.
            # Wall-modeled approach (y+~30–100) keeps cells <30k and runs in time.
            size_near_wall = max(size_near_wall, D / 10)
            size_bulk = D / 3

            if wall_tags:
                dist_f = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_f, "SurfacesList", wall_tags)
                th_f = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(th_f, "InField", dist_f)
                gmsh.model.mesh.field.setNumber(th_f, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(th_f, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(th_f, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(th_f, "DistMax", R * 0.3)
                gmsh.model.mesh.field.setAsBackgroundMesh(th_f)

            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)
            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_bfs(self, params: CFDParams, msh_file: Path):
        """Backward-facing step: L-shaped 2D domain extruded.

        Good procedure (from PROJECT_STATUS):
        - Distance/Threshold refinement on all walls
        - Extra Point refinement at step corner (separation origin)
        - size_near_wall = max(pol.first_cell_height, H/30)
        - size_bulk = H/6
        """
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("bfs")
        self._current_params = params

        H = params.width            # downstream channel height
        Hin = H / 2                 # upstream channel height (= step height h)
        L_up = max(params.length / 6, Hin * 2)   # upstream: at least 2h
        L_dn = params.length - L_up
        depth = 0.001

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H / 20
        size_near_wall = max(size_near_wall, H / 30)   # clamp: prevent 100k+ cells
        size_corner    = size_near_wall * 0.5           # finest near step corner
        size_bulk      = max(H / 6, size_near_wall * 4)

        try:
            # L-shaped domain: upstream channel + downstream expansion
            lc_bg = size_bulk   # background size for point placement
            pts = [
                gmsh.model.occ.addPoint(-L_up, Hin, 0, lc_bg),  # p1: inlet top
                gmsh.model.occ.addPoint(0,     Hin, 0, size_corner),  # p2: step top
                gmsh.model.occ.addPoint(0,     0,   0, size_corner),  # p3: step bottom (corner)
                gmsh.model.occ.addPoint(L_dn,  0,   0, lc_bg),   # p4: outlet bottom
                gmsh.model.occ.addPoint(L_dn,  H,   0, lc_bg),   # p5: outlet top
                gmsh.model.occ.addPoint(-L_up, H,   0, lc_bg),   # p6: inlet top-wall
            ]
            lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                     for i in range(len(pts))]
            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()

            ext = gmsh.model.occ.extrude(
                [(2, surf)], 0, 0, depth, numElements=[1], recombine=True
            )
            gmsh.model.occ.synchronize()

            vol_tag = ext[1][1]
            front_tag = ext[0][1]
            lat_surfs = [(d, t) for d, t in ext[2:] if d == 2]

            inlet_tags, outlet_tags, wall_tags = [], [], []
            tol = min(L_up, L_dn) * 0.02
            for _, tag in lat_surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0])
                xmid = (bb[0] + bb[3]) / 2
                if xspan < tol and xmid < -L_up + tol:
                    inlet_tags.append(tag)
                elif xspan < tol and xmid > L_dn - tol:
                    outlet_tags.append(tag)
                else:
                    wall_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf, front_tag], name="frontAndBack")
            if inlet_tags:
                gmsh.model.addPhysicalGroup(2, inlet_tags, name="inlet")
            if outlet_tags:
                gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if wall_tags:
                gmsh.model.addPhysicalGroup(2, wall_tags, name="walls")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            # Distance/Threshold: refine near all walls + tightest at step corner
            if wall_tags:
                dist_f = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_f, "SurfacesList", wall_tags)
                th_f = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(th_f, "InField", dist_f)
                gmsh.model.mesh.field.setNumber(th_f, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(th_f, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(th_f, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(th_f, "DistMax", Hin * 0.4)

                # Extra Point field at step corner (0, 0) for extra refinement
                corner_f = gmsh.model.mesh.field.add("Ball")
                gmsh.model.mesh.field.setNumber(corner_f, "Radius", Hin * 0.3)
                gmsh.model.mesh.field.setNumber(corner_f, "VIn", size_corner)
                gmsh.model.mesh.field.setNumber(corner_f, "VOut", size_bulk)
                gmsh.model.mesh.field.setNumber(corner_f, "XCenter", 0.0)
                gmsh.model.mesh.field.setNumber(corner_f, "YCenter", 0.0)
                gmsh.model.mesh.field.setNumber(corner_f, "ZCenter", depth / 2)

                # Min field takes finest value at each point
                min_f = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(min_f, "FieldsList", [th_f, corner_f])
                gmsh.model.mesh.field.setAsBackgroundMesh(min_f)

            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_corner)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)
            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_airfoil_box(self, params: CFDParams, msh_file: Path):
        """NACA 4-digit airfoil (parses code from prompt; defaults to 0012)
        in a rectangular far-field box."""
        import gmsh
        import math
        import re
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("airfoil")

        chord = params.length if params.length <= 1.0 else 1.0
        aoa = params.angle_of_attack or 0.0
        depth = 0.001

        # Far-field box: 20c x 20c centered at c/4
        box_x0, box_x1 = -10 * chord, 20 * chord
        box_y0, box_y1 = -10 * chord, 10 * chord

        # Parse NACA 4-digit code from extraction_notes / refined params.
        # Format MPXX → m=M%, p=P/10 chord, t=XX/100 thickness.
        m = re.search(r"naca\s*(\d{4})", (params.extraction_notes or "").lower())
        if m:
            code = m.group(1)
        else:
            code = "0012"
        m_camber = int(code[0]) / 100.0          # max camber as fraction of chord
        p_camber = int(code[1]) / 10.0           # location of max camber
        t_thick  = int(code[2:4]) / 100.0        # max thickness as fraction of chord

        # NACA 4-digit coordinates with cosine-spaced x stations
        n_pts = 80
        pts_upper, pts_lower = [], []
        for i in range(n_pts + 1):
            x = chord * (1 - math.cos(math.pi * i / n_pts)) / 2
            xc = x / chord
            yt = 5 * t_thick * chord * (
                0.2969 * math.sqrt(xc)
                - 0.1260 * xc
                - 0.3516 * xc ** 2
                + 0.2843 * xc ** 3
                - 0.1015 * xc ** 4
            )
            # Mean camber line (zero for symmetric NACA 00XX)
            if m_camber > 0 and 0 < p_camber < 1:
                if xc < p_camber:
                    yc = chord * m_camber / (p_camber ** 2) * (
                        2 * p_camber * xc - xc ** 2)
                    dy = m_camber / (p_camber ** 2) * (2 * p_camber - 2 * xc)
                else:
                    yc = chord * m_camber / ((1 - p_camber) ** 2) * (
                        (1 - 2 * p_camber) + 2 * p_camber * xc - xc ** 2)
                    dy = m_camber / ((1 - p_camber) ** 2) * (2 * p_camber - 2 * xc)
                theta = math.atan(dy)
                xu_loc = x - yt * math.sin(theta)
                yu_loc = yc + yt * math.cos(theta)
                xl_loc = x + yt * math.sin(theta)
                yl_loc = yc - yt * math.cos(theta)
            else:
                xu_loc, yu_loc = x, yt
                xl_loc, yl_loc = x, -yt
            # Rotate by AoA
            rad = math.radians(aoa)
            cosA, sinA = math.cos(rad), math.sin(rad)
            pts_upper.append(( xu_loc * cosA + yu_loc * sinA,
                              -xu_loc * sinA + yu_loc * cosA))
            pts_lower.append(( xl_loc * cosA + yl_loc * sinA,
                              -xl_loc * sinA + yl_loc * cosA))

        try:
            # Outer box
            box = gmsh.model.occ.addRectangle(
                box_x0, box_y0, 0, box_x1 - box_x0, box_y1 - box_y0
            )
            gmsh.model.occ.synchronize()

            # Airfoil spline
            foil_pts = []
            for x, y in pts_upper:
                foil_pts.append(gmsh.model.occ.addPoint(x, y, 0, chord / 60))
            for x, y in reversed(pts_lower[1:-1]):
                foil_pts.append(gmsh.model.occ.addPoint(x, y, 0, chord / 60))
            spline = gmsh.model.occ.addSpline(foil_pts + [foil_pts[0]])
            foil_loop = gmsh.model.occ.addCurveLoop([spline])
            foil_surf = gmsh.model.occ.addPlaneSurface([foil_loop])
            gmsh.model.occ.synchronize()

            domain, _ = gmsh.model.occ.cut([(2, box)], [(2, foil_surf)])
            gmsh.model.occ.synchronize()

            surfs = gmsh.model.occ.getEntities(2)
            surf_tag = surfs[0][1] if surfs else 1

            ext = gmsh.model.occ.extrude(
                [(2, surf_tag)], 0, 0, depth, numElements=[1], recombine=True
            )
            gmsh.model.occ.synchronize()

            vol_tag = ext[1][1]
            front_tag = ext[0][1]
            lat_surfs = [(d, t) for d, t in ext[2:] if d == 2]

            inlet_tags, outlet_tags, airfoil_tags, ff_tags = [], [], [], []
            for _, tag in lat_surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0])
                yspan = abs(bb[4] - bb[1])
                if xspan < 1e-6 and abs(bb[0] - box_x0) < 0.01:
                    inlet_tags.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - box_x1) < 0.01:
                    outlet_tags.append(tag)
                elif yspan < 1e-6:
                    ff_tags.append(tag)
                else:
                    airfoil_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf_tag, front_tag], name="frontAndBack")
            gmsh.model.addPhysicalGroup(2, inlet_tags + ff_tags, name="freestream")
            gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if airfoil_tags:
                gmsh.model.addPhysicalGroup(2, airfoil_tags, name="airfoil")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            # Refinement near airfoil — policy-aware sizing
            # Good procedure: size_near_wall from NumericalPolicy, clamped to chord/50
            # to prevent 100k+ cells at high Re
            if airfoil_tags:
                pol = getattr(self, "_policy", None)
                size_surface = pol.first_cell_height if pol else chord / 50
                size_surface = max(size_surface, chord / 50)  # clamp: no finer than chord/50
                size_wake    = chord / 20                      # wake region
                size_farfield = chord * 2                      # far-field coarse

                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", airfoil_tags)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_surface)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_farfield)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", chord / 20)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", chord * 5)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_surface)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_farfield)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_wedge(self, params: CFDParams, msh_file: Path):
        """5-degree axisymmetric wedge for OpenFOAM wedge BC.

        Good procedure:
        - Revolve xy-plane rectangle 5° around x-axis
        - front face (θ=0): all z ≈ 0  → distinguished by zmax < ε
        - back  face (θ=5°): has z > 0 → zmax > ε
        - Distance/Threshold wall refinement near r=R boundary
        """
        import gmsh
        import math
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("wedge")
        self._current_params = params

        R = (params.diameter or params.width) / 2.0
        L = params.length

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else R / 10
        size_near_wall = max(size_near_wall, R / 10)   # clamp
        size_bulk = max(R / 3, size_near_wall * 4)

        try:
            # Rectangular rz-profile in xy-plane (z=0)
            p1 = gmsh.model.occ.addPoint(0, 0, 0, size_near_wall)
            p2 = gmsh.model.occ.addPoint(L, 0, 0, size_near_wall)
            p3 = gmsh.model.occ.addPoint(L, R, 0, size_near_wall)
            p4 = gmsh.model.occ.addPoint(0, R, 0, size_near_wall)
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addLine(p2, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addLine(p4, p1)
            loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()

            # Revolve 5 degrees around x-axis
            ext = gmsh.model.occ.revolve(
                [(2, surf)], 0, 0, 0, 1, 0, 0, math.radians(5)
            )
            gmsh.model.occ.synchronize()

            vol_tag = ext[1][1]
            surfs = gmsh.model.occ.getEntities(2)
            inlet_tags, outlet_tags, wall_tags, axis_tags = [], [], [], []
            front_tags, back_tags = [], []

            z_tol = R * math.sin(math.radians(5)) * 0.01   # ≈ 0.1% of wedge z-extent

            for _, tag in surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xmin, ymin, zmin = bb[0], bb[1], bb[2]
                xmax, ymax, zmax = bb[3], bb[4], bb[5]
                xspan = abs(xmax - xmin)
                ymax_abs = max(abs(ymin), abs(ymax))

                if xspan < 1e-6 and xmin < 0.01:
                    inlet_tags.append(tag)
                elif xspan < 1e-6 and xmin > L - 0.01:
                    outlet_tags.append(tag)
                elif ymax_abs < z_tol * 10:
                    # Near y=0 and z=0 → axis
                    axis_tags.append(tag)
                elif ymax_abs > R * 0.9:
                    # Outer wall surface (r ≈ R)
                    wall_tags.append(tag)
                elif zmax < z_tol:
                    # All z ≈ 0 → front face (θ=0, original xy-plane)
                    front_tags.append(tag)
                else:
                    # Has z > 0 → back face (θ=5°, revolved face)
                    back_tags.append(tag)

            if inlet_tags:
                gmsh.model.addPhysicalGroup(2, inlet_tags, name="inlet")
            if outlet_tags:
                gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if wall_tags:
                gmsh.model.addPhysicalGroup(2, wall_tags, name="wall")
            if axis_tags:
                gmsh.model.addPhysicalGroup(2, axis_tags, name="axis")
            if front_tags:
                gmsh.model.addPhysicalGroup(2, front_tags, name="front")
            if back_tags:
                gmsh.model.addPhysicalGroup(2, back_tags, name="back")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            # Distance/Threshold wall refinement (pipe near-wall sizing)
            if wall_tags:
                dist_f = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(dist_f, "SurfacesList", wall_tags)
                th_f = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(th_f, "InField", dist_f)
                gmsh.model.mesh.field.setNumber(th_f, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(th_f, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(th_f, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(th_f, "DistMax", R * 0.3)
                gmsh.model.mesh.field.setAsBackgroundMesh(th_f)

            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)
            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    # ------------------------------------------------------------------ #
    #  New builders (NACA 4-digit handled via _build_airfoil_box parsing
    #  the prompt; the four below cover periodic hill, S-bend, diffuser,
    #  and 3D sphere-in-box). All 2D builders extrude a thin slab so the
    #  case is OpenFOAM-2D-compatible (frontAndBack empty patches).
    # ------------------------------------------------------------------ #

    def _build_pehill(self, params: CFDParams, msh_file: Path):
        """Wu / Breuer periodic-hill benchmark, 2D extruded.
        Hill profile follows the polynomial pieces of Mellen et al.
        Patches: bottomWall (curved), topWall, inlet, outlet, frontAndBack.
        """
        import gmsh
        import math
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("pehill")
        self._current_params = params

        H = params.width or 1.0           # hill height (canonical = 1)
        Lx = max(params.length, 9.0 * H)  # one period (canonical Lx = 9 H)
        Ly = 3.035 * H                    # canonical channel height
        depth = 0.001

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H / 30
        size_near_wall = max(size_near_wall, H / 40)
        size_bulk = max(H / 6, size_near_wall * 4)

        # Mellen periodic-hill profile, x in [0, 9H]; bottom y(x).
        def hill_y(x):
            xs = abs(x) / H
            if xs > 9.0:
                return 0.0
            mirror = xs > 4.5
            if mirror:
                xs = 9.0 - xs
            if xs < 0.3214:
                y = 1.0 + 0.0 * xs - 5.231e-3 * xs**2 + 0.0 * xs**3 + 0.0 * xs**4 + 0.0 * xs**5
                y = min(1.0, 1.0 - 0.5 * xs**2)  # smoother top of hill
            elif xs < 0.5:
                y = 0.8955 + 3.484e-2 * xs - 3.629 * xs**2 + 6.749 * xs**3 - 4.452 * xs**4 + 0.0 * xs**5
            elif xs < 0.7143:
                y = 0.9213 - 0.5908 * xs + 4.077 * xs**2 - 12.28 * xs**3 + 14.06 * xs**4 - 5.706 * xs**5
            elif xs < 1.071:
                y = 1.445 - 2.16 * xs - 7.394e-3 * xs**2 + 4.477 * xs**3 - 4.014 * xs**4 + 1.165 * xs**5
            elif xs < 1.429:
                y = 0.638 - 0.4995 * xs - 5.272 * xs**2 + 11.26 * xs**3 - 8.165 * xs**4 + 2.075 * xs**5
            elif xs < 1.929:
                y = 1.6 - 1.965 * xs - 0.1019 * xs**2 + 1.405 * xs**3 - 1.018 * xs**4 + 0.2606 * xs**5
            else:
                y = 0.0
            return max(0.0, H * y)

        try:
            # Sample points along the hill curve
            n_hill = 80
            hill_pts = []
            for i in range(n_hill + 1):
                x = (i / n_hill) * Lx
                y = hill_y(x)
                hill_pts.append(gmsh.model.occ.addPoint(x, y, 0, size_near_wall))
            # Spline along hill
            hill_curve = gmsh.model.occ.addSpline(hill_pts)
            # Top wall (flat)
            tl = gmsh.model.occ.addPoint(0,  Ly, 0, size_bulk)
            tr = gmsh.model.occ.addPoint(Lx, Ly, 0, size_bulk)
            # Vertical sides
            lin_left  = gmsh.model.occ.addLine(hill_pts[0],  tl)
            lin_top   = gmsh.model.occ.addLine(tl, tr)
            lin_right = gmsh.model.occ.addLine(tr, hill_pts[-1])
            loop = gmsh.model.occ.addCurveLoop([hill_curve, lin_right, -lin_top, -lin_left])
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()

            ext = gmsh.model.occ.extrude(
                [(2, surf)], 0, 0, depth, numElements=[1], recombine=True)
            gmsh.model.occ.synchronize()
            vol_tag = ext[1][1]
            front_tag = ext[0][1]
            lat_surfs = [(d, t) for d, t in ext[2:] if d == 2]

            inlet_tags, outlet_tags, top_tags, bottom_tags = [], [], [], []
            for _, tag in lat_surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xmid, ymid = (bb[0]+bb[3])/2, (bb[1]+bb[4])/2
                xspan, yspan = abs(bb[3]-bb[0]), abs(bb[4]-bb[1])
                if xspan < 1e-6 and abs(bb[0]) < 1e-6:
                    inlet_tags.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - Lx) < 1e-6:
                    outlet_tags.append(tag)
                elif yspan < 1e-6 and abs(bb[1] - Ly) < 1e-6:
                    top_tags.append(tag)
                else:
                    bottom_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf, front_tag], name="frontAndBack")
            if inlet_tags:   gmsh.model.addPhysicalGroup(2, inlet_tags,   name="inlet")
            if outlet_tags:  gmsh.model.addPhysicalGroup(2, outlet_tags,  name="outlet")
            if top_tags:     gmsh.model.addPhysicalGroup(2, top_tags,     name="topWall")
            if bottom_tags:  gmsh.model.addPhysicalGroup(2, bottom_tags,  name="bottomWall")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            wall_all = top_tags + bottom_tags
            if wall_all:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", wall_all)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", H * 0.4)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_sbend(self, params: CFDParams, msh_file: Path):
        """S-bend duct, 2D extruded. The centreline follows a single sine
        period so the shape is a true S (one up-bump + one down-bump).
        Patches: inlet, outlet, walls, frontAndBack.
        """
        import gmsh
        import math
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("sbend")
        self._current_params = params

        H = params.width or 0.05      # duct half-height
        L_total = max(params.length, 8 * H)
        L_in  = L_total * 0.15
        L_out = L_total * 0.15
        L_bend = L_total - L_in - L_out
        A_amp  = 4 * H                 # vertical excursion of the S
        depth = 0.001

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H / 12
        size_near_wall = max(size_near_wall, H / 15)
        size_bulk = max(H / 4, size_near_wall * 4)

        try:
            # Centreline: straight inlet leg, sinusoidal S, straight outlet leg.
            # Single sine period crosses zero three times → one full "S".
            n_seg = 60
            cl = []
            for i in range(n_seg + 1):
                x = i / n_seg * L_total
                if x <= L_in:
                    y = 0.0
                elif x >= L_total - L_out:
                    y = 0.0
                else:
                    t = (x - L_in) / L_bend       # t in [0, 1]
                    y = A_amp * math.sin(2 * math.pi * t)
                cl.append((x, y))

            # Build top + bottom walls offset by ±H from centreline.
            # Use forward / backward differences at endpoints (central inside)
            # so the local tangent is well-defined and the offset is correct.
            top, bot = [], []
            for i in range(len(cl)):
                if i == 0:
                    dx = cl[1][0] - cl[0][0]
                    dy = cl[1][1] - cl[0][1]
                elif i == len(cl) - 1:
                    dx = cl[-1][0] - cl[-2][0]
                    dy = cl[-1][1] - cl[-2][1]
                else:
                    dx = cl[i + 1][0] - cl[i - 1][0]
                    dy = cl[i + 1][1] - cl[i - 1][1]
                norm = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / norm, dx / norm
                top.append(gmsh.model.occ.addPoint(cl[i][0] + nx * H,
                                                    cl[i][1] + ny * H, 0,
                                                    size_near_wall))
                bot.append(gmsh.model.occ.addPoint(cl[i][0] - nx * H,
                                                    cl[i][1] - ny * H, 0,
                                                    size_near_wall))

            top_curve = gmsh.model.occ.addSpline(top)
            bot_curve = gmsh.model.occ.addSpline(bot)
            inlet_line  = gmsh.model.occ.addLine(top[0], bot[0])
            outlet_line = gmsh.model.occ.addLine(bot[-1], top[-1])
            loop = gmsh.model.occ.addCurveLoop([top_curve, outlet_line,
                                                  -bot_curve, -inlet_line])
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()

            ext = gmsh.model.occ.extrude(
                [(2, surf)], 0, 0, depth, numElements=[1], recombine=True)
            gmsh.model.occ.synchronize()
            vol_tag = ext[1][1]; front_tag = ext[0][1]
            lat_surfs = [(d, t) for d, t in ext[2:] if d == 2]

            # Tag boundaries by examining 1D entities — inlet/outlet are short,
            # top/bottom walls are the long splines.
            inlet_tags, outlet_tags, wall_tags = [], [], []
            for _, tag in lat_surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0])
                if xspan < H * 0.5:
                    if (bb[0] + bb[3]) / 2 < L_total * 0.5:
                        inlet_tags.append(tag)
                    else:
                        outlet_tags.append(tag)
                else:
                    wall_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf, front_tag], name="frontAndBack")
            if inlet_tags:  gmsh.model.addPhysicalGroup(2, inlet_tags,  name="inlet")
            if outlet_tags: gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if wall_tags:   gmsh.model.addPhysicalGroup(2, wall_tags,   name="walls")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            if wall_tags:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", wall_tags)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", H * 0.5)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_diffuser(self, params: CFDParams, msh_file: Path):
        """2D planar diffuser: rectangular inlet that linearly expands to a
        wider outlet. Patches: inlet, outlet, walls, frontAndBack.
        Uses an area-ratio of 2 by default (sets divergence half-angle ≈ 4°).
        """
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("diffuser")
        self._current_params = params

        L = max(params.length, 0.5)
        H_in  = params.width or 0.1     # inlet half-height
        H_out = 2.0 * H_in              # outlet half-height (AR=2)
        depth = 0.001

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H_in / 12
        size_near_wall = max(size_near_wall, H_in / 15)
        size_bulk = max(H_in / 4, size_near_wall * 4)

        try:
            p1 = gmsh.model.occ.addPoint(0,  H_in,  0, size_near_wall)
            p2 = gmsh.model.occ.addPoint(L,  H_out, 0, size_near_wall)
            p3 = gmsh.model.occ.addPoint(L, -H_out, 0, size_near_wall)
            p4 = gmsh.model.occ.addPoint(0, -H_in,  0, size_near_wall)
            l1 = gmsh.model.occ.addLine(p1, p2)
            l2 = gmsh.model.occ.addLine(p2, p3)
            l3 = gmsh.model.occ.addLine(p3, p4)
            l4 = gmsh.model.occ.addLine(p4, p1)
            loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()

            ext = gmsh.model.occ.extrude(
                [(2, surf)], 0, 0, depth, numElements=[1], recombine=True)
            gmsh.model.occ.synchronize()
            vol_tag = ext[1][1]; front_tag = ext[0][1]
            lat_surfs = [(d, t) for d, t in ext[2:] if d == 2]

            inlet_tags, outlet_tags, wall_tags = [], [], []
            for _, tag in lat_surfs:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0])
                xmid = (bb[0] + bb[3]) / 2
                if xspan < 1e-6 and xmid < L * 0.05:
                    inlet_tags.append(tag)
                elif xspan < 1e-6 and xmid > L * 0.95:
                    outlet_tags.append(tag)
                else:
                    wall_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf, front_tag], name="frontAndBack")
            if inlet_tags:  gmsh.model.addPhysicalGroup(2, inlet_tags,  name="inlet")
            if outlet_tags: gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if wall_tags:   gmsh.model.addPhysicalGroup(2, wall_tags,   name="walls")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            if wall_tags:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", wall_tags)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", H_in * 0.4)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_sphere_3d(self, params: CFDParams, msh_file: Path):
        """3D sphere in a rectangular far-field box.
        Patches: inlet, outlet, sphere, walls (top/bottom/side far-field).
        Domain: 10D upstream, 25D downstream, ±10D crossflow each side.
        """
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("sphere3d")
        self._current_params = params

        D = params.diameter or params.width or 0.1
        R = D / 2.0
        x0, x1 = -10 * D, 25 * D
        half  = 10 * D

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else D / 20
        size_near_wall = max(size_near_wall, D / 30)
        size_bulk = max(D, size_near_wall * 8)

        try:
            box = gmsh.model.occ.addBox(x0, -half, -half, x1 - x0, 2 * half, 2 * half)
            sph = gmsh.model.occ.addSphere(0, 0, 0, R)
            domain, _ = gmsh.model.occ.cut([(3, box)], [(3, sph)])
            gmsh.model.occ.synchronize()

            vol_tag = domain[0][1]
            surfs = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)

            inlet_tags, outlet_tags, sph_tags, ff_tags = [], [], [], []
            for d, tag in surfs:
                if d != 2:
                    continue
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0])
                yspan = abs(bb[4] - bb[1])
                zspan = abs(bb[5] - bb[2])
                xmid = (bb[0] + bb[3]) / 2
                ymid = (bb[1] + bb[4]) / 2
                zmid = (bb[2] + bb[5]) / 2
                # Sphere surface: small extents and centred near origin
                if (xspan < D * 1.1 and yspan < D * 1.1 and zspan < D * 1.1
                        and abs(xmid) < D and abs(ymid) < D and abs(zmid) < D):
                    sph_tags.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - x0) < 0.01:
                    inlet_tags.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - x1) < 0.01:
                    outlet_tags.append(tag)
                else:
                    ff_tags.append(tag)

            if inlet_tags:  gmsh.model.addPhysicalGroup(2, inlet_tags,  name="inlet")
            if outlet_tags: gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if sph_tags:    gmsh.model.addPhysicalGroup(2, sph_tags,    name="sphere")
            if ff_tags:     gmsh.model.addPhysicalGroup(2, ff_tags,     name="walls")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            if sph_tags:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", sph_tags)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", D * 4)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    # ------------------------------------------------------------------ #
    #  Second batch: Ahmed body, multi-hill, T-junction, CD nozzle, elbow
    # ------------------------------------------------------------------ #

    def _build_ahmed_body(self, params: CFDParams, msh_file: Path):
        """Ahmed body — automotive bluff-body benchmark, 3D.
        Side profile: front-rounded box of length L, height H, with a
        slanted rear face at angle phi (defaults 25°). Body sits at the
        floor inside a wind-tunnel domain (3L upstream, 8L downstream,
        ±2.5L spanwise, 4H high).
        Patches: inlet, outlet, body, ground, top, sides.
        """
        import gmsh, math
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("ahmed")
        self._current_params = params

        L = params.length or 1.044
        W = params.width or 0.389
        H = params.diameter or 0.288  # use diameter slot for body height
        slant_deg = params.angle_of_attack or 25.0
        L_slant = 0.222
        H_rear = max(0.05, H - L_slant * math.tan(math.radians(slant_deg)))

        # Domain
        x0, x1 = -3 * L, 8 * L
        y0, y1 = 0.0, 4 * H
        z0, z1 = -2.5 * L, 2.5 * L

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H / 25
        size_near_wall = max(size_near_wall, H / 30)
        size_bulk = max(H / 2, size_near_wall * 6)

        try:
            # 2D side profile of body (xy plane, z=0)
            pts = [(0, 0), (L, 0), (L, H_rear),
                   (L - L_slant, H), (0, H)]
            gmsh_pts = [gmsh.model.occ.addPoint(x, y, -W / 2, size_near_wall)
                         for x, y in pts]
            lines = [gmsh.model.occ.addLine(gmsh_pts[i], gmsh_pts[(i + 1) % 5])
                      for i in range(5)]
            loop = gmsh.model.occ.addCurveLoop(lines)
            face = gmsh.model.occ.addPlaneSurface([loop])
            ext = gmsh.model.occ.extrude([(2, face)], 0, 0, W,
                                           numElements=[1], recombine=False)
            body_vol = ext[1][1]
            gmsh.model.occ.synchronize()

            # Domain box
            domain = gmsh.model.occ.addBox(x0, y0, z0,
                                             x1 - x0, y1 - y0, z1 - z0)
            gmsh.model.occ.synchronize()

            cut, _ = gmsh.model.occ.cut([(3, domain)], [(3, body_vol)])
            gmsh.model.occ.synchronize()
            vol_tag = cut[0][1]

            inlet_tags, outlet_tags, top_tags, ground_tags = [], [], [], []
            side_tags, body_tags = [], []
            for d, tag in gmsh.model.getBoundary([(3, vol_tag)], oriented=False):
                if d != 2:
                    continue
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0]); yspan = abs(bb[4] - bb[1]); zspan = abs(bb[5] - bb[2])
                xmid = (bb[0] + bb[3]) / 2; ymid = (bb[1] + bb[4]) / 2; zmid = (bb[2] + bb[5]) / 2
                if   xspan < 1e-3 and abs(bb[0] - x0) < 1e-3: inlet_tags.append(tag)
                elif xspan < 1e-3 and abs(bb[0] - x1) < 1e-3: outlet_tags.append(tag)
                elif yspan < 1e-3 and abs(bb[1] - y1) < 1e-3: top_tags.append(tag)
                elif yspan < 1e-3 and abs(bb[1] - y0) < 1e-3: ground_tags.append(tag)
                elif zspan < 1e-3 and (abs(bb[2] - z0) < 1e-3 or abs(bb[2] - z1) < 1e-3):
                    side_tags.append(tag)
                else:
                    body_tags.append(tag)

            if inlet_tags:  gmsh.model.addPhysicalGroup(2, inlet_tags,  name="inlet")
            if outlet_tags: gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if top_tags:    gmsh.model.addPhysicalGroup(2, top_tags,    name="top")
            if ground_tags: gmsh.model.addPhysicalGroup(2, ground_tags, name="ground")
            if side_tags:   gmsh.model.addPhysicalGroup(2, side_tags,   name="sides")
            if body_tags:   gmsh.model.addPhysicalGroup(2, body_tags,   name="body")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            if body_tags:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", body_tags)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", L * 0.5)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_multi_hill(self, params: CFDParams, msh_file: Path):
        """Multiple periodic hills in series — typically 3 or 4 Mellen-shaped
        hills along x. Patches: bottomWall, topWall, inlet, outlet, frontAndBack.
        """
        import gmsh, math
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("multihill")
        self._current_params = params

        H = params.width or 1.0           # hill height
        n_hills = 3
        Lx_each = 9.0 * H                 # one period
        Lx = max(params.length, n_hills * Lx_each)
        Ly = 3.035 * H
        depth = 0.001

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H / 25
        size_near_wall = max(size_near_wall, H / 30)
        size_bulk = max(H / 5, size_near_wall * 4)

        # Mellen profile for one period (x in [0, 9H], shifted/repeated).
        def hill_y_one(x):
            xs = abs(x) / H
            if xs > 9.0: return 0.0
            mirror = xs > 4.5
            if mirror: xs = 9.0 - xs
            if xs < 0.3214:
                y = min(1.0, 1.0 - 0.5 * xs**2)
            elif xs < 0.5:
                y = 0.8955 + 3.484e-2*xs - 3.629*xs**2 + 6.749*xs**3 - 4.452*xs**4
            elif xs < 0.7143:
                y = 0.9213 - 0.5908*xs + 4.077*xs**2 - 12.28*xs**3 + 14.06*xs**4 - 5.706*xs**5
            elif xs < 1.071:
                y = 1.445 - 2.16*xs - 7.394e-3*xs**2 + 4.477*xs**3 - 4.014*xs**4 + 1.165*xs**5
            elif xs < 1.429:
                y = 0.638 - 0.4995*xs - 5.272*xs**2 + 11.26*xs**3 - 8.165*xs**4 + 2.075*xs**5
            elif xs < 1.929:
                y = 1.6 - 1.965*xs - 0.1019*xs**2 + 1.405*xs**3 - 1.018*xs**4 + 0.2606*xs**5
            else:
                y = 0.0
            return max(0.0, H * y)

        try:
            n_pts = 80 * n_hills
            hill_pts = []
            for i in range(n_pts + 1):
                x = (i / n_pts) * Lx
                # x mod one period
                x_mod = x % Lx_each
                y = hill_y_one(x_mod)
                hill_pts.append(gmsh.model.occ.addPoint(x, y, 0, size_near_wall))
            hill_curve = gmsh.model.occ.addSpline(hill_pts)
            tl = gmsh.model.occ.addPoint(0,  Ly, 0, size_bulk)
            tr = gmsh.model.occ.addPoint(Lx, Ly, 0, size_bulk)
            lin_left  = gmsh.model.occ.addLine(hill_pts[0],  tl)
            lin_top   = gmsh.model.occ.addLine(tl, tr)
            lin_right = gmsh.model.occ.addLine(tr, hill_pts[-1])
            loop = gmsh.model.occ.addCurveLoop([hill_curve, lin_right, -lin_top, -lin_left])
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()
            ext = gmsh.model.occ.extrude([(2, surf)], 0, 0, depth,
                                           numElements=[1], recombine=True)
            gmsh.model.occ.synchronize()
            vol_tag = ext[1][1]; front_tag = ext[0][1]

            inlet_tags, outlet_tags, top_tags, bot_tags = [], [], [], []
            for d, tag in [(d, t) for d, t in ext[2:] if d == 2]:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0]); yspan = abs(bb[4] - bb[1])
                if   xspan < 1e-6 and bb[0] < 1e-6:           inlet_tags.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - Lx) < 1e-6: outlet_tags.append(tag)
                elif yspan < 1e-6 and abs(bb[1] - Ly) < 1e-6: top_tags.append(tag)
                else:                                          bot_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf, front_tag], name="frontAndBack")
            if inlet_tags:  gmsh.model.addPhysicalGroup(2, inlet_tags,  name="inlet")
            if outlet_tags: gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if top_tags:    gmsh.model.addPhysicalGroup(2, top_tags,    name="topWall")
            if bot_tags:    gmsh.model.addPhysicalGroup(2, bot_tags,    name="bottomWall")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            wall_all = top_tags + bot_tags
            if wall_all:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", wall_all)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", H * 0.4)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_tjunction(self, params: CFDParams, msh_file: Path):
        """T-junction: horizontal main channel with a perpendicular branch
        rising from its midpoint. Patches: mainInlet, branchInlet, outlet,
        walls, frontAndBack.
        """
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("tjunc")
        self._current_params = params

        L = max(params.length, 0.6)
        H = params.width or 0.05            # main half-height
        Lb = L * 0.4                        # branch length
        depth = 0.001

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H / 12
        size_near_wall = max(size_near_wall, H / 15)
        size_bulk = max(H / 4, size_near_wall * 4)

        # Polygon walks the boundary clockwise:
        # main inlet bottom-left → main bottom → outlet bottom-right
        # → outlet top-right → branch right wall → branch top → branch left
        # → main top-right of branch → main top → main top-left → close
        x_branch_l = L * 0.5 - H
        x_branch_r = L * 0.5 + H
        y_branch_top = H + Lb
        try:
            pts = [
                (0,            -H),               # main inlet bottom
                (L,            -H),               # outlet bottom
                (L,             H),               # outlet top
                (x_branch_r,    H),               # right corner
                (x_branch_r,    y_branch_top),    # branch top-right
                (x_branch_l,    y_branch_top),    # branch top-left
                (x_branch_l,    H),               # left corner
                (0,             H),               # main top-left
            ]
            gpts = [gmsh.model.occ.addPoint(x, y, 0, size_near_wall) for x, y in pts]
            lines = [gmsh.model.occ.addLine(gpts[i], gpts[(i + 1) % len(gpts)])
                      for i in range(len(gpts))]
            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()
            ext = gmsh.model.occ.extrude([(2, surf)], 0, 0, depth,
                                          numElements=[1], recombine=True)
            gmsh.model.occ.synchronize()
            vol_tag = ext[1][1]; front_tag = ext[0][1]

            main_inlet, branch_inlet, outlet, wall_tags = [], [], [], []
            for d, tag in [(d, t) for d, t in ext[2:] if d == 2]:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0]); yspan = abs(bb[4] - bb[1])
                xmid = (bb[0] + bb[3]) / 2; ymid = (bb[1] + bb[4]) / 2
                if   xspan < 1e-6 and bb[0] < 1e-6:               main_inlet.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - L) < 1e-6:      outlet.append(tag)
                elif yspan < 1e-6 and abs(bb[1] - y_branch_top) < 1e-6: branch_inlet.append(tag)
                else:                                              wall_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf, front_tag], name="frontAndBack")
            if main_inlet:   gmsh.model.addPhysicalGroup(2, main_inlet,   name="mainInlet")
            if branch_inlet: gmsh.model.addPhysicalGroup(2, branch_inlet, name="branchInlet")
            if outlet:       gmsh.model.addPhysicalGroup(2, outlet,       name="outlet")
            if wall_tags:    gmsh.model.addPhysicalGroup(2, wall_tags,    name="walls")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            if wall_tags:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", wall_tags)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", H * 0.5)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_cd_nozzle(self, params: CFDParams, msh_file: Path):
        """Convergent-divergent (de Laval) nozzle, 2D extruded.
        Symmetric profile: inlet half-height H_in → throat H_th → outlet H_out.
        Patches: inlet, outlet, walls, frontAndBack.
        """
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("cdnozzle")
        self._current_params = params

        L      = max(params.length, 0.5)
        H_in   = params.width or 0.10
        H_th   = H_in * 0.4
        H_out  = H_in * 0.7
        x_th   = L * 0.4
        depth  = 0.001

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H_th / 10
        size_near_wall = max(size_near_wall, H_th / 12)
        size_bulk = max(H_in / 4, size_near_wall * 4)

        try:
            # Symmetric profile (top + mirror bottom)
            top = [(0, H_in), (x_th, H_th), (L, H_out)]
            bot = [(L, -H_out), (x_th, -H_th), (0, -H_in)]
            poly = top + bot
            gpts = [gmsh.model.occ.addPoint(x, y, 0, size_near_wall) for x, y in poly]
            lines = [gmsh.model.occ.addLine(gpts[i], gpts[(i + 1) % len(gpts)])
                      for i in range(len(gpts))]
            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()
            ext = gmsh.model.occ.extrude([(2, surf)], 0, 0, depth,
                                          numElements=[1], recombine=True)
            gmsh.model.occ.synchronize()
            vol_tag = ext[1][1]; front_tag = ext[0][1]

            inlet_tags, outlet_tags, wall_tags = [], [], []
            for d, tag in [(d, t) for d, t in ext[2:] if d == 2]:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0])
                if xspan < 1e-6 and bb[0] < 1e-6:           inlet_tags.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - L) < 1e-6: outlet_tags.append(tag)
                else:                                         wall_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf, front_tag], name="frontAndBack")
            if inlet_tags:  gmsh.model.addPhysicalGroup(2, inlet_tags,  name="inlet")
            if outlet_tags: gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if wall_tags:   gmsh.model.addPhysicalGroup(2, wall_tags,   name="walls")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            if wall_tags:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", wall_tags)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", H_th * 0.6)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    def _build_elbow(self, params: CFDParams, msh_file: Path):
        """90° elbow duct, 2D extruded. Two equal straight legs joined at a
        right-angle bend with a square inner corner.
        Patches: inlet, outlet, walls, frontAndBack.
        """
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.model.add("elbow")
        self._current_params = params

        H = params.width or 0.05
        Leg = max(params.length / 2, 4 * H)
        depth = 0.001

        pol = getattr(self, "_policy", None)
        size_near_wall = pol.first_cell_height if pol else H / 12
        size_near_wall = max(size_near_wall, H / 15)
        size_bulk = max(H / 4, size_near_wall * 4)

        # L-shaped polygon (clockwise):
        # inlet at bottom of vertical leg (x=0, y=-Leg → y=2H)
        # bend → horizontal leg (y in [H,2H], x in [0, Leg])
        # outlet at right end (x=Leg, y in [H,2H])
        try:
            pts = [
                (0,   -Leg),     # inlet bottom
                (2*H, -Leg),     # inlet bottom-right
                (2*H,  H),       # inner corner
                (Leg,  H),       # outlet bottom
                (Leg,  2*H),     # outlet top
                (0,    2*H),     # bend top-outer
            ]
            gpts = [gmsh.model.occ.addPoint(x, y, 0, size_near_wall) for x, y in pts]
            lines = [gmsh.model.occ.addLine(gpts[i], gpts[(i + 1) % len(gpts)])
                      for i in range(len(gpts))]
            loop = gmsh.model.occ.addCurveLoop(lines)
            surf = gmsh.model.occ.addPlaneSurface([loop])
            gmsh.model.occ.synchronize()
            ext = gmsh.model.occ.extrude([(2, surf)], 0, 0, depth,
                                          numElements=[1], recombine=True)
            gmsh.model.occ.synchronize()
            vol_tag = ext[1][1]; front_tag = ext[0][1]

            inlet_tags, outlet_tags, wall_tags = [], [], []
            for d, tag in [(d, t) for d, t in ext[2:] if d == 2]:
                bb = gmsh.model.occ.getBoundingBox(2, tag)
                xspan = abs(bb[3] - bb[0]); yspan = abs(bb[4] - bb[1])
                if yspan < 1e-6 and abs(bb[1] + Leg) < 1e-6:    inlet_tags.append(tag)
                elif xspan < 1e-6 and abs(bb[0] - Leg) < 1e-6:  outlet_tags.append(tag)
                else:                                            wall_tags.append(tag)

            gmsh.model.addPhysicalGroup(2, [surf, front_tag], name="frontAndBack")
            if inlet_tags:  gmsh.model.addPhysicalGroup(2, inlet_tags,  name="inlet")
            if outlet_tags: gmsh.model.addPhysicalGroup(2, outlet_tags, name="outlet")
            if wall_tags:   gmsh.model.addPhysicalGroup(2, wall_tags,   name="walls")
            gmsh.model.addPhysicalGroup(3, [vol_tag], name="fluid")

            if wall_tags:
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "SurfacesList", wall_tags)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", size_near_wall)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", size_bulk)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", H * 0.5)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin", size_near_wall)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_bulk)

            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_file))
        finally:
            gmsh.finalize()

    # ------------------------------------------------------------------ #
    #  Post-processing
    # ------------------------------------------------------------------ #

    def _run_gmsh_to_foam(self, msh_file: Path, case_dir: Path):
        # gmshToFoam requires system/controlDict to exist
        (case_dir / "system").mkdir(parents=True, exist_ok=True)
        ctrl = case_dir / "system" / "controlDict"
        if not ctrl.exists():
            ctrl.write_text(
                "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class dictionary;\n"
                "    object controlDict;\n}\n"
                "application     simpleFoam;\n"
                "startFrom       startTime;\nstartTime       0;\n"
                "stopAt          endTime;\nendTime         1;\n"
                "deltaT          1;\n"
                "writeControl    timeStep;\nwriteInterval   1;\n"
                "writeFrequency  1;\n"
                "purgeWrite      0;\nwriteFormat     ascii;\nwritePrecision  6;\n"
                "writeCompression off;\ntimeFormat      general;\ntimePrecision   6;\n"
                "runTimeModifiable true;\n"
            )

        cmd = (
            f"bash -c 'source {OPENFOAM_BASHRC} && "
            f"gmshToFoam {msh_file} -case {case_dir}'"
        )
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"gmshToFoam failed:\n{result.stderr[-2000:]}")

    def _fix_boundary_file(self, case_dir: Path):
        boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
        if not boundary_file.exists():
            return

        text = boundary_file.read_text()
        lines = text.splitlines()
        new_lines = []
        current_patch = ""

        for line in lines:
            stripped = line.strip()
            # Detect patch name lines (single word before opening brace context)
            if stripped and not stripped.startswith("//") and not stripped.startswith("("):
                words = stripped.split()
                if len(words) == 1 and words[0] not in ("{", "}", "FoamFile"):
                    current_patch = words[0]

            # Fix type lines
            if stripped.startswith("type") and current_patch:
                correct_type = _patch_type(current_patch)
                indent = line[: len(line) - len(line.lstrip())]
                new_lines.append(f"{indent}type            {correct_type};")
                continue

            new_lines.append(line)

        boundary_file.write_text("\n".join(new_lines))
