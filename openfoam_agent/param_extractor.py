from __future__ import annotations

import json

from .schemas import CFDParams, FlowRegime, TurbulenceModel

_SYSTEM_PROMPT = """You are a CFD parameter extraction system. Given a CFD problem specification, extract ALL parameters as a JSON object matching the provided schema exactly.

OUTPUT
- Output ONLY valid JSON. No prose, no markdown, no code fences.
- Include EVERY field in the schema — never omit a field.

NUMERIC FIDELITY (very important)
- Preserve every numeric value the user states VERBATIM. Convert units to SI
  (mm → m, cm → m, °C if temperature, etc.) but never round or substitute.
- NEVER invent a velocity, length, viscosity, density, or Reynolds number
  that the user did not state. If a value is missing, derive it from the
  others using physical relations below; if you cannot, leave it 0/None and
  the post-processor will default it.
- If multiple values together over-determine the problem (e.g. user gives
  Re, U, L, ν that are mutually inconsistent), trust the order
  Re > L > ν > U: re-derive U from Re·ν/L_char.

REYNOLDS-NUMBER CONSISTENCY
- Always set fields so that  Re = U · L_char / ν  is exactly satisfied.
- L_char (the Reynolds characteristic length) depends on geometry:
    pipe, cylinder           → diameter
    airfoil                  → chord (store the chord in the `length` field)
    lid_driven_cavity, box,
    channel, wedge,
    backward_facing_step     → streamwise length
- If the user gives Re and L_char but no U: set U = Re · ν / L_char.
- If the user gives U and L_char but no Re: set Re = U · L_char / ν.

FLUID PROPERTIES (use canonical values when a fluid is named)
    water       ν = 1.0e-6  m²/s,  ρ = 998   kg/m³
    seawater    ν = 1.05e-6 m²/s,  ρ = 1025  kg/m³
    glycerine   ν = 1.18e-3 m²/s,  ρ = 1260  kg/m³
    engine oil  ν = 5.0e-4  m²/s,  ρ = 870   kg/m³
    air         ν = 1.5e-5  m²/s,  ρ = 1.225 kg/m³  (default if no fluid named)
    nitrogen    ν = 1.6e-5  m²/s,  ρ = 1.165 kg/m³
    CO2         ν = 8.0e-6  m²/s,  ρ = 1.842 kg/m³
- If the user names a fluid, USE these values. Do not default to air when a
  different fluid is named.

GEOMETRY
- Available `geometry_type` enum values:
    lid_driven_cavity, channel, box, cylinder, pipe, backward_facing_step,
    airfoil, wedge, sphere, periodic_hill, s_bend, diffuser,
    ahmed_body, multi_hill, t_junction, cd_nozzle, elbow, custom.
- Pick `periodic_hill` for "periodic hill", "Wu hill", "Mellen hill",
  "Breuer hill", or any flow with a wavy / hill-shaped lower wall and
  cyclic streamwise direction.
- Pick `multi_hill` for "multiple periodic hills", "three hills in series",
  "row of hills", "hill train".
- Pick `s_bend` for "S-bend", "S-shaped duct", "U-bend", "double-curve duct".
- Pick `elbow` for "elbow", "90 degree bend", "right-angle bend",
  "L-shaped duct", "right-angle duct".
- Pick `t_junction` for "T-junction", "T-piece", "tee", "branching duct",
  "manifold with branch", "main + branch flow".
- Pick `cd_nozzle` for "convergent-divergent nozzle", "de Laval nozzle",
  "throat", "rocket nozzle", "Laval".
- Pick `diffuser` for "diffuser", "expanding duct", "nozzle expansion",
  "subsonic diffuser" (use rhoSimpleFoam if compressible).
- Pick `sphere` for "sphere", "ball", "spherical body", "sphere drag",
  "flow over a sphere".
- Pick `ahmed_body` for "Ahmed body", "Ahmed reference model", or any
  automotive bluff-body wind-tunnel geometry. Store the slant angle in
  `angle_of_attack`.
- Pick `airfoil` for any "NACAxxxx", "NACA xxxx", "hydrofoil", "wing
  section", "blade profile". Store the chord in `length`. Put the literal
  4-digit NACA code (e.g. "NACA 4412") into `extraction_notes` so the
  geometry builder can read it.
- For 2D cases: height = 0.001 (thin empty-BC domain), is_3d = false.
- For 3D pipes/cylinders: height = diameter (cross-section).
- For pipes/cylinders: diameter = characteristic dimension D.
- For cubical / square geometries (e.g. "1m square", "2m × 2m", "0.5m cube"):
  width = length, and if 3D height = length too.

REGIMES & SOLVERS
- flow_regime: laminar if Re<2300, transitional if 2300-4000, turbulent if Re>4000.
- turbulence_model: "laminar" if laminar, "kOmegaSST" if turbulent (or "kEpsilon" if user asks).
- MULTIPHASE (VOF): set is_multiphase=true for dam break, wave, water column collapse, free-surface, or two-fluid flows.
- COMPRESSIBLE: set is_compressible=true for Mach > 0.3 or supersonic/transonic/compressible flows.
- HEAT TRANSFER: set has_heat_transfer=true for natural convection, buoyancy, heated walls, or temperature fields.
- TRANSIENT: set is_transient=true for unsteady, time-varying, vortex shedding, dam break, or dynamic simulations.
- For steady-state: end_time = 1000, is_transient = false."""


def extract(llm, refined_prompt: str, original_prompt: str | None = None) -> CFDParams:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    schema = CFDParams.model_json_schema()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": refined_prompt},
    ]
    outputs = llm.chat(messages, sampling_params=sampling)
    raw_json = outputs[0].outputs[0].text

    # Parse raw JSON to detect which fields were actually emitted by the LLM
    # (vs. relying on Pydantic defaults which lose that distinction)
    raw_dict = json.loads(raw_json)
    return _post_validate_raw(raw_dict, prompt=refined_prompt,
                                original_prompt=original_prompt)


_GEOMETRY_DIMS: dict[str, dict] = {
    "lid_driven_cavity": {"length": 1.0, "width": 1.0, "height": 0.001},
    "channel": {"length": 10.0, "width": 1.0, "height": 0.001},
    "box": {"length": 1.0, "width": 1.0, "height": 1.0},
    "cylinder": {"length": 10.0, "width": 4.0, "height": 0.001},
    "pipe": {"length": 0.5, "width": 0.05, "height": 0.05},
    "backward_facing_step": {"length": 0.1, "width": 0.05, "height": 0.001},
    "airfoil": {"length": 4.0, "width": 4.0, "height": 0.001},
    "sphere": {"length": 2.0, "width": 2.0, "height": 2.0},
    "wedge": {"length": 1.0, "width": 0.05, "height": 0.001},
    # New parametric templates
    "periodic_hill": {"length": 9.0,   "width": 1.0,   "height": 0.001},
    "s_bend":        {"length": 1.0,   "width": 0.05,  "height": 0.001},
    "diffuser":      {"length": 1.0,   "width": 0.1,   "height": 0.001},
    "ahmed_body":    {"length": 1.044, "width": 0.389, "height": 0.288},  # canonical Ahmed
    "multi_hill":    {"length": 27.0,  "width": 1.0,   "height": 0.001},  # ~3 × 9H periods
    "t_junction":    {"length": 1.0,   "width": 0.05,  "height": 0.001},
    "cd_nozzle":     {"length": 0.5,   "width": 0.10,  "height": 0.001},
    "elbow":         {"length": 0.4,   "width": 0.05,  "height": 0.001},
}


_MULTIPHASE_KEYWORDS = ("dam break", "dam-break", "dambreak", "water column",
                         "free surface", "free-surface", "wave channel",
                         "wave tank", "interfoam", "vof", "two-phase",
                         "two phase", "multiphase", "flood", "sloshing",
                         "water collapse", "alpha.water", "bubble column")
_COMPRESSIBLE_KEYWORDS = ("mach", " ma=", " ma ", "supersonic", "transonic",
                           "subsonic", "compressible", "rhosimplefoam",
                           "rhopimplefoam", "ideal gas", "shock")
_HEAT_KEYWORDS = ("natural convection", "buoyancy", "buoyant", "buoyantsimplefoam",
                   "buoyantpimplefoam", "heated wall", "heated bottom",
                   "heated top", "heat transfer", "rayleigh-benard",
                   "rayleigh benard", "rb convection", "differentially heated",
                   "hot wall", "cold wall", "hot radiator", "cold window")
_TRANSIENT_KEYWORDS = ("unsteady", "transient", "time-varying", "time-dependent",
                        "time-resolved", "time dependent", "time resolved",
                        "vortex shedding", "dam break", "dam-break", "dambreak",
                        "water column", "wave channel", "wave tank", "dynamic",
                        "pulsatile", "pulsating", "pulsing", "impulsive",
                        "impulsively", "starting", "oscillating", "icofoam",
                        "pimplefoam", "rhopimplefoam", "buoyantpimplefoam",
                        "urans")

# Fluid library at standard conditions (kinematic viscosity m²/s, density kg/m³).
# Order matters: more specific keywords come first so "glycerine" matches before "water".
_FLUIDS: list[tuple[tuple[str, ...], float, float]] = [
    (("glycerine", "glycerin"), 1.18e-3, 1260.0),
    (("engine oil", "motor oil", "lubricating oil"), 5.0e-4, 870.0),
    (("seawater", "sea water"), 1.05e-6, 1025.0),
    (("water",), 1.0e-6, 998.0),
    (("nitrogen", "n2 gas"), 1.6e-5, 1.165),
    (("hydrogen", "h2 gas"), 1.06e-4, 0.0899),
    (("helium", "he gas"), 1.18e-4, 0.1786),
    (("co2", "carbon dioxide"), 8.0e-6, 1.842),
    (("steam",), 2.0e-5, 0.598),
    (("air",), 1.5e-5, 1.225),
]


def _fluid_from_prompt(prompt_lower: str) -> tuple[float | None, float | None]:
    """Return (nu, rho) inferred from fluid keyword in the prompt, or (None, None)
    if no fluid keyword matches. Used as a fallback when the LLM omits or
    mis-extracts the kinematic viscosity."""
    for kws, nu, rho in _FLUIDS:
        if any(kw in prompt_lower for kw in kws):
            return nu, rho
    return None, None


import re as _re_module

# A single regex that captures a Reynolds-number specification in the user's
# prompt — matches 're=1000', 're = 1e6', 'Re 5000', 'Reynolds number 1.5e5',
# 'Re of 100k', etc. We use the same regex for both the boolean check and
# value extraction so they can never disagree.
_RE_NUM = r"([0-9]+(?:\.[0-9]+)?(?:\s*[xX]\s*10\^?[0-9]+|e[+-]?[0-9]+)?)"
_RE_PATTERNS = [
    _re_module.compile(rf"re(?:ynolds)?(?:\s+number)?\s*(?:[=:]|of|is|~|≈|approx)?\s*{_RE_NUM}\s*([km])?\b",
                       _re_module.IGNORECASE),
]


def _prompt_has_explicit_re(prompt_lower: str) -> bool:
    """True if the prompt mentions Reynolds number explicitly."""
    return _extract_re_from_prompt(prompt_lower) is not None


def _extract_re_from_prompt(prompt_lower: str) -> float | None:
    """Best-effort regex extraction of a Reynolds-number value when the LLM
    drops it. Recognises 'Re=1000', 'Re 1e6', 'Reynolds number 5000', 'Re 100k', etc."""
    for pat in _RE_PATTERNS:
        m = pat.search(prompt_lower)
        if not m:
            continue
        tok, suffix = m.group(1), (m.group(2) or "").lower()
        # Normalise '1.5 x 10^5' / '1.5x10^5' style
        tok = tok.replace(" ", "").replace("x10^", "e").replace("X10^", "e")
        try:
            v = float(tok)
        except ValueError:
            continue
        if suffix == "k":
            v *= 1e3
        elif suffix == "m":
            v *= 1e6
        if v > 0:
            return v
    return None


_LEN_NUM = r"([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)"
_LEN_PATTERNS = {
    "chord": _re_module.compile(rf"chord(?:\s*length)?\s*(?:[=:]|of|is|~|≈)?\s*{_LEN_NUM}\s*(mm|cm|m|meters?)\b",
                                  _re_module.IGNORECASE),
    "diameter": _re_module.compile(rf"(?:diameter|d\s*[=:]|d\s*of)\s*[=:]?\s*{_LEN_NUM}\s*(mm|cm|m|meters?)\b",
                                     _re_module.IGNORECASE),
}


def _to_meters(value: float, unit: str) -> float:
    u = unit.lower()
    if u == "mm":
        return value * 1e-3
    if u == "cm":
        return value * 1e-2
    return value


def _extract_dim_from_prompt(prompt_lower: str, key: str) -> float | None:
    """Pull a length value (in metres) for 'chord' or 'diameter' out of the prompt."""
    pat = _LEN_PATTERNS.get(key)
    if pat is None:
        return None
    m = pat.search(prompt_lower)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return _to_meters(v, m.group(2))


def _prompt_implies_square(prompt_lower: str) -> bool:
    """True if the prompt describes a square / cubical geometry whose width
    should match its length."""
    return any(kw in prompt_lower for kw in
               (" square", "square ", "cubic", "cubical", "n × n", "x x", "1:1"))


def _post_validate_raw(raw: dict, prompt: str = "", original_prompt: str | None = None) -> CFDParams:
    geom = raw.get("geometry_type", "box")
    if hasattr(geom, "value"):
        geom = geom.value
    dims = _GEOMETRY_DIMS.get(geom, {"length": 1.0, "width": 1.0, "height": 0.001})
    is_3d = raw.get("is_3d", False)
    diameter = raw.get("diameter")

    # --- Geometry-specific dimension defaults for missing/zero fields ---
    def _get(field: str, fallback: float) -> float:
        v = raw.get(field)
        return v if (v is not None and v > 0) else fallback

    length = _get("length", dims["length"])

    # Prompt-regex overrides for chord (airfoil) and diameter (pipe/cylinder).
    # The refiner sometimes drops or rewrites these numeric specs; always
    # trust the user's literal value when one is given.
    _user_prompt_for_dims = (original_prompt if original_prompt else prompt).lower()
    if geom == "airfoil":
        chord = _extract_dim_from_prompt(_user_prompt_for_dims, "chord")
        if chord and chord > 0:
            length = chord  # airfoil convention: length stores chord
    if geom in ("pipe", "cylinder"):
        d_user = _extract_dim_from_prompt(_user_prompt_for_dims, "diameter")
        if d_user and d_user > 0:
            diameter = d_user
    outlet_pressure = raw.get("outlet_pressure", 0.0)

    # Keyword heuristics — use ONLY the user's original prompt as the source
    # of truth for keyword detection. The refiner may invent numbers (U=1 m/s
    # default for Re-only prompts) and the LLM's `extraction_notes` field may
    # contain explanatory text like "considering multiphase…" that would
    # leak unwanted keywords into the multiphase / compressible / heat /
    # transient classifiers below.
    user_prompt = original_prompt if original_prompt else prompt
    prompt_lower = user_prompt.lower()

    # Fluid properties: prefer LLM-extracted nu/rho when present and physical;
    # otherwise infer from a fluid keyword in the prompt; otherwise default to air.
    raw_nu = raw.get("kinematic_viscosity")
    raw_rho = raw.get("density")
    fluid_nu, fluid_rho = _fluid_from_prompt(prompt_lower)
    if raw_nu and raw_nu > 0:
        # Sanity check: if a fluid keyword is named and the LLM's nu is off by
        # >5× from the canonical value, override (LLM hallucinated air for water).
        if fluid_nu is not None and not (0.2 * fluid_nu <= raw_nu <= 5 * fluid_nu):
            nu = fluid_nu
        else:
            nu = raw_nu
    else:
        nu = fluid_nu if fluid_nu is not None else 1.5e-5
    if raw_rho and raw_rho > 0:
        if fluid_rho is not None and not (0.2 * fluid_rho <= raw_rho <= 5 * fluid_rho):
            density = fluid_rho
        else:
            density = raw_rho
    else:
        density = fluid_rho if fluid_rho is not None else 1.225
    # Physics flags — keyword-decisive policy. The fine-tuned LLM tends to
    # over-predict is_transient and is_multiphase on prompts whose only cue
    # is the geometry. To stop those false positives, we make the user's
    # prompt the source of truth: a flag is True only if a corresponding
    # keyword is present, OR the LLM said True AND there is at least a weak
    # corroborating signal in the prompt (e.g. an "ideal gas" hint for
    # compressibility, or a numbered Mach value).
    kw_multiphase    = any(kw in prompt_lower for kw in _MULTIPHASE_KEYWORDS)
    kw_compressible  = any(kw in prompt_lower for kw in _COMPRESSIBLE_KEYWORDS)
    kw_heat          = any(kw in prompt_lower for kw in _HEAT_KEYWORDS)
    kw_transient     = any(kw in prompt_lower for kw in _TRANSIENT_KEYWORDS)
    # Explicit "steady" wording is a strong negative signal for transience.
    explicit_steady  = any(kw in prompt_lower for kw in
                            ("steady-state", "steady state", " steady ",
                             "steady,", "steady.", "steady;"))

    is_multiphase    = kw_multiphase
    is_compressible  = kw_compressible
    has_heat_transfer = kw_heat
    is_transient     = kw_transient or (is_multiphase and not explicit_steady)
    if explicit_steady:
        is_transient = False

    # end_time: use LLM value if present, else short default for transient (10s),
    # long default for steady-state (1000 iterations)
    raw_end_time = raw.get("end_time")
    if raw_end_time and raw_end_time > 0:
        end_time = raw_end_time
    elif is_transient:
        end_time = 10.0
    else:
        end_time = 1000.0

    # Width: use diameter if available for pipes/cylinders
    if diameter and diameter > 0:
        if geom == "pipe":
            width = diameter
        elif geom == "cylinder":
            # domain width = 8*D for flow-over-cylinder benchmark
            width = _get("width", max(8 * diameter, dims["width"]))
        else:
            width = _get("width", dims["width"])
    else:
        width = _get("width", dims["width"])

    # If the prompt describes a square / cubical geometry, force width = length
    # (and depth = length for 3D), regardless of what the LLM emitted.
    if _prompt_implies_square(prompt_lower) and geom in (
            "lid_driven_cavity", "box"):
        width = length
        if is_3d:
            # height set below; for now keep length
            pass

    # Height
    if is_3d:
        if geom == "pipe" and diameter:
            height = _get("height", diameter)
        else:
            height = _get("height", dims["height"])
    else:
        height = 0.001

    # Reynolds number — accept LLM value, then fallback to regex-extract from
    # the prompt, then leave as None for downstream to derive from U,L,nu.
    re = raw.get("reynolds_number")
    if re is not None and re <= 0:
        re = None
    if re is None:
        re = _extract_re_from_prompt(prompt_lower)

    # Choose the characteristic length used in Re = U·L_char/ν.
    #   pipes / cylinders → diameter
    #   airfoils          → chord (stored in `length`)
    #   cavities / boxes / channels / wedges → streamwise length
    if geom in ("pipe", "cylinder") and diameter:
        char_len = diameter
    elif geom == "airfoil":
        char_len = length  # chord
    elif geom in ("lid_driven_cavity", "box", "channel", "wedge",
                   "backward_facing_step"):
        char_len = length
    else:
        char_len = diameter or length

    # Inlet velocity. The LLM (and especially the refiner) commonly emits a
    # default U=1.0 even when the user only specified Re. Detect whether the
    # *original* user prompt mentions a velocity at all; if it doesn't but
    # mentions Re, derive U = Re * nu / L unconditionally.
    raw_u = raw.get("inlet_velocity")
    user_mentions_velocity = any(
        kw in prompt_lower for kw in
        ("m/s", "m / s", "u=", "u =", "velocity ", "velocity=", "freestream",
         "inlet velocity", "lid velocity", "speed of"))
    explicit_re = re is not None and _prompt_has_explicit_re(prompt_lower)
    if explicit_re and not user_mentions_velocity and char_len > 0:
        inlet_velocity = re * nu / char_len
    elif raw_u and raw_u > 0:
        inlet_velocity = raw_u
    elif re is not None and char_len > 0:
        inlet_velocity = re * nu / char_len
    else:
        inlet_velocity = _get("inlet_velocity", 1.0)

    # Compute Re if still missing, using the same characteristic length.
    if re is None and char_len > 0:
        re = inlet_velocity * char_len / nu

    # Flow regime from Re
    if re < 2300:
        flow_regime = FlowRegime.LAMINAR
    elif re > 4000:
        flow_regime = FlowRegime.TURBULENT
    else:
        flow_regime = FlowRegime.TRANSITIONAL

    # Turbulence model
    raw_turb = raw.get("turbulence_model", "")
    if flow_regime == FlowRegime.LAMINAR:
        turbulence_model = TurbulenceModel.LAMINAR
    elif raw_turb in ("kEpsilon",):
        turbulence_model = TurbulenceModel.K_EPSILON
    else:
        turbulence_model = TurbulenceModel.K_OMEGA_SST

    # Diameter for pipe/cylinder
    if diameter is None or diameter <= 0:
        if geom in ("pipe", "cylinder"):
            diameter = width if geom == "pipe" else _get("diameter", dims.get("width", 0.1))

    data = {
        "geometry_type": geom,
        "is_3d": is_3d,
        "length": length,
        "width": width,
        "height": height,
        "diameter": diameter,
        "reynolds_number": re,
        "inlet_velocity": inlet_velocity,
        "outlet_pressure": outlet_pressure,
        "kinematic_viscosity": nu,
        "density": density,
        "flow_regime": flow_regime,
        "turbulence_model": turbulence_model,
        "is_transient": is_transient,
        "is_compressible": is_compressible,
        "has_heat_transfer": has_heat_transfer,
        "is_multiphase": is_multiphase,
        "end_time": end_time,
        "angle_of_attack": raw.get("angle_of_attack"),
        "extraction_notes": raw.get("extraction_notes", ""),
    }

    return CFDParams.model_validate(data)
