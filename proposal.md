Large Language Models and Physics-AI for Fluid Dynamic Simulations

Large language models in combination with physics-inspired AI models (NVIDIA PhysicsNemo) can modernize fluid dynamics simulations by making them more automated, interpretable, and faster. 

Recent developments such as MetaOpenFOAM, OpenFOAM-GPT, and Foam-Agent, where natural language interfaces are integrated with physics-aware models to streamline simulation setup, accelerate analysis, and enable interactive exploration of complex flows. By combining LLMs with tools like MetaOpenFOAM, OpenFOAM-GPT, and Foam-Agent, simulations can be set up, optimized, and analyzed with minimal human intervention while remaining physically accurate. This integrated pipelines accelerates complex multi-physics and CFD computations, reduces manual intervention, and enables interactive exploration of flow phenomena (such as in crystal growth), bridging natural language reasoning with high-fidelity, physics-informed modeling.

Computational Fluid Dynamics plays a critical role in domains such as materials science and cryogenics, yet its complexity often places it beyond the reach of many data scientists. In this talk, we present how large language models (LLMs) and physics-inspired AI models (PhysicsNemo) can be combined to make fluid dynamics simulations more accessible, automated, and interactive. By integrating AI agents and natural language interfaces with open-source tools such as MetaOpenFOAM, OpenFOAM-GPT, and Foam-Agent, users can configure, run, and analyze simulations using Python-centric, reproducible workflows.

We discuss in detail two use cases we have been working on —Silicon single crystal growth and cryogenic flow simulations—implemented using state-of-the-art physics-inspired machine learning models. For crystal growth, PhysicsNemo-based Physics-Informed Neural Networks (PINNs) and modified PINNs with adaptive constraint weighting are implemented in JAX and PyTorch to solve tightly coupled Navier–Stokes, energy, and species transport equations. Operator-learning models such as PI-GANO are employed to learn solution operators across varying boundary conditions and process parameters, enabling rapid inference for large-scale design exploration. For cryogenic simulations on complex geometries, MeshGraphNet architectures are used to learn spatiotemporal flow dynamics on unstructured meshes, integrating physical priors directly into graph message-passing layers.

Training these models at scale requires substantial compute, which we address using PhysicsNemo’s built-in parallelization strategies, including data and model parallelism across multiple GPUs. Distributed training is orchestrated via PyTorch Distributed Data Parallel (DDP) and JAX, enabling efficient learning from large OpenFOAM-generated datasets and accelerating convergence for high-resolution simulations. we intend to demonstrates how modern scientific machine learning stacks can be combined with physics solvers to build scalable, reproducible, and performance-oriented simulation pipelines.

To Summarize, we cover in this talk the aspects related to 

LLM-aided CFD workflows (such as MetaOpenFOAM, OpenFOAM-GPT, and Foam-Agent)
Physics-inspired ML models: PINNs, modified PINNs, PI-GANO, and MeshGraphNet
Use cases: Crystal growth and cryogenic flow simulation use cases
Model development in PyTorch and JAX with PhysicsNemo
Multi-GPU training via PhysicsNemo parallelization, PyTorch DDP
Results and Discussions



we are developing these methods in context of understanding silicon crystal growth simulations and Gas Phase Condensation in Cryogenic Vessels and estimating boil-off gas (BOG) and boil-off rate (BOR) in cryogenic liquid storage tanks (Liquid hydrogen) using multiphase and  thermal CFD  

More details are in these repos
https://github.com/adytiaa/Gas_Condens_Cryogenic_surrogate
https://github.com/adytiaa/cryogenic_vessel_sim
https://github.com/adytiaa/CG_PI_GA_AI
