from __future__ import annotations

from .schemas import RefinedPrompt

_SYSTEM_PROMPT = """You are an expert CFD consultant. Your job is to take vague simulation requests and rewrite them as precise, complete problem specifications that an OpenFOAM engineer can use directly.

For EVERY prompt you must specify ALL of the following:
- Geometry type and exact dimensions (length, width, height/diameter) in meters
- Fluid properties: kinematic viscosity nu in m^2/s, density rho in kg/m^3
- Inlet velocity magnitude in m/s and direction
- Reynolds number Re = U * L_characteristic / nu
- Flow regime: laminar (Re < 2300), transitional (2300-4000), turbulent (Re > 4000)
- Steady-state or transient
- 2D or 3D
- Appropriate turbulence model: laminar, kOmegaSST, or kEpsilon

Default values to use when not specified:
- Fluid: air (nu=1.5e-5 m^2/s, rho=1.225 kg/m^3) unless "water" is mentioned
- Water: nu=1e-6 m^2/s, rho=1000 kg/m^3
- 2D domain depth: 0.001 m (single-cell empty BC)
- Lid-driven cavity default: 1m x 1m, U_lid=1 m/s
- Pipe default: D=0.05m, L=0.5m
- Cylinder default: D=0.1m, domain 2m x 0.4m
- Channel default: L=10m, H=0.1m
- Turbulence model: laminar if Re<2300, kOmegaSST if Re>=2300

For steady vs transient:
- Lid-driven cavity at low Re: steady (simpleFoam)
- Cylinder flow Re>50: transient (pimpleFoam)
- Pipe/channel turbulent: steady (simpleFoam)
- Explicitly transient if user says "time-dependent", "unsteady", or "transient"

Output ONLY the refined problem statement as a single descriptive paragraph. No headings, no bullet points, no explanations."""


def refine(llm, raw_prompt: str) -> RefinedPrompt:
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.3, max_tokens=512)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": raw_prompt},
    ]
    outputs = llm.chat(messages, sampling_params=sampling)
    refined_text = outputs[0].outputs[0].text.strip()

    return RefinedPrompt(
        original=raw_prompt,
        refined=refined_text,
        added_context="Defaults applied where not specified",
        detected_ambiguities=[],
    )
