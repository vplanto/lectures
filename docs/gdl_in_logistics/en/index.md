# Course: Geometric Deep Learning in Logistics
## Topic: Isomorphism of Routing Problems and Protein Folding (AlphaFold 2)

> 🇺🇦 [Українська версія](../index.md) | 🇬🇧 English version

**Author:** Vitalii Platonov
**Version:** 1.0
**Status:** Draft
**Audience:** Technical students, R&D Engineers

---

### Course Annotation

This series of lectures and practical workshops aims to combine classical Operations Research theory with modern Geometric Deep Learning approaches. We examine how AlphaFold 2 architectural solutions, developed for predicting protein 3D structure, can be adapted to solve NP-complete routing problems (VRP, TSP) at scales unattainable by classical exact algorithms.

### Teaching Kit (NMK)

| | |
|---|---|
| **Methodology** | [Course trajectory](./methodology.md) |
| **Sources** | [Primary-source register](../sources.md) |
| **Declaration** | [Academic integrity](../../en/DISCLAIMER.md) |


---

### Course Structure (Navigation)

#### Course Introduction
Overview of scaling problems, mathematical foundation of Geometric Deep Learning, and biological analogy.

* **[00_geometric_deep_learning_intro.md](./en/00_geometric_deep_learning_intro.md)**
    * **Topic:** Geometric Deep Learning in Logistics (Introduction).
    * **Content:** Classical algorithm scaling problem. Curse of dimensionality. Geometric Deep Learning through group theory. AlphaFold 2 as engineering solution. Isomorphism of protein folding and routing problems. Engineering implementation (memory, parallelism, throughput).

#### Block 1: Problem Foundation
Classical formulation, mathematical model, and applicability limits of existing solutions.

* **[01_classical_transport_problem.md](./en/01_classical_transport_problem.md)**
    * **Topic:** Classical formulation of transport problems.
    * **Content:** History (Monge-Kantorovich), mathematical formalization (objective function, constraints), problem types (VRP, TSP, Assignment Problem).
    
* **[02_algorithm_limitations.md](./en/02_algorithm_limitations.md)**
    * **Topic:** Classical algorithm limitations.
    * **Content:** Overview of Dijkstra, A*, Branch & Bound. Curse of dimensionality concept. Why exact methods die at $N>20$, and heuristics get stuck in local minima.

#### Block 2: Biological Analogy and AlphaFold 2
Deconstructing architecture that changed science through graph theory lens.

* **[03_protein_folding_problem.md](./en/03_protein_folding_problem.md)**
    * **Topic:** Protein folding problem formulation.
    * **Content:** Protein as graph. Levinthal's paradox. Energy landscape. Analogy: amino acid = client, polypeptide chain = route.

* **[04_alphafold2_architecture.md](./en/04_alphafold2_architecture.md)**
    * **Topic:** AlphaFold 2 deconstruction.
    * **Content:** Geometry Tower, Evoformer, Invariant Point Attention (IPA). How attention mechanism reconstructs spatial structure from evolutionary information (MSA).

* **[05_np_completeness_isomorphism.md](./en/05_np_completeness_isomorphism.md)**
    * **Topic:** AlphaFold 2 and NP-complete problem isomorphism.
    * **Content:** Proof that protein folding and finding optimal route are energy/cost minimization problems on graphs. Why AF2 geometric approach works better than combinatorial enumeration.

#### Block 3: Synthesis and Adaptation
Transforming biological model into logistics solver.

* **[06_tsp_via_af2_geometry.md](./en/06_tsp_via_af2_geometry.md)**
    * **Topic:** TSP formulation under AlphaFold architecture.
    * **Content:** Entity mapping: MSA -> Traffic history, Pair Representation -> Distance matrix. Replacing physical constraints (atoms) with logistics (time windows, fuel). Agent interpretation: "Atom as Agent" — IPA/Geodesic Attention as autonomous agent sensor system (couriers, pedestrians, vehicles). 4D space geometry: time windows as spacetime corridors. Dynamic flow simulation: Recycling as time-steps in agent modeling — each iteration predicts where agents will be after Δt. Identifying "bottlenecks" through energy landscape: energy funnel, high "energy" zones (conflicts, congestion), prediction through Geodesic Attention. Emergent behavior: from protein folding to traffic jams — mathematical analogy between atom and pedestrian evacuation, agent simulation with 10,000 RPS throughput, congestion analysis as mathematical obstacles in 4D spacetime.

* **[07_implementation_methodology.md](./en/07_implementation_methodology.md)**
    * **Topic:** Implementation methodology (Deep Dive).
    * **Content:** Step-by-step Evoformer adaptation guide. Triangular Multiplicative Update for logistics. Loss functions for VRP.

#### Block 4: Workshop (Practical)
From theory to working code with AI assistants.

* **[08_ai_assisted_development.md](./en/08_ai_assisted_development.md)**
    * **Topic:** Prompt engineering for Cursor/Claude.
    * **Content:** Specific system prompts for  generating complex mathematical code (PyTorch/JAX). How to make LLM write correct tensor operations.

* **[09_synthetic_benchmark.md](./en/09_synthetic_benchmark.md)**
    * **Topic:** Creating synthetic example.
    * **Content:** Dataset generation (city coordinates, cost matrix, constraints). Data preparation in tensor format for model.

* **[10_solution_comparison.md](./en/10_solution_comparison.md)**
    * **Topic:** Final solution and validation.
    * **Content:** Manual calculation (Reference). Software solution execution. Results comparison (accuracy vs time). Conclusions.

#### Additional Seminars
Deep dives into key mathematical concepts.

* **[11_uncertainty_weighted_loss_seminar.md](./en/11_uncertainty_weighted_loss_seminar.md)**
    * **Topic:** Uncertainty-Weighted Loss: Elegant mathematical solution for balancing Loss components.
    * **Content:** "Eyeballing" coefficient problem. Uncertainty-Weighted Loss derivation from Bayesian optimization. Detailed mathematical analysis (gradients, optimality, asymptotics). PyTorch practical implementation. Comparison with alternative methods. Mathematical exercises and proofs.

* **[12_emergent_behavior.md](./en/12_emergent_behavior.md)**
    * **Topic:** Emergent Behavior: From protein folding to traffic jams.
    * **Content:** Mathematical analogy between protein folding and pedestrian evacuation. Agent simulation with real-time trajectory prediction (10,000 RPS throughput). Congestion analysis as mathematical obstacles in 4D spacetime where Time Penalty becomes critical. Correspondence to 2nd year "Applied Mathematics" program.

* **[13_evacuation_workshop.md](./en/13_evacuation_workshop.md)**
    * **Topic:** Workshop: Stadium evacuation — A* vs GDL.
    * **Content:** Detailed comparison of classical A* and GDL approaches for evacuating 10,000 pedestrians from stadium. A* shows shortest path but creates congestion. GDL (like AlphaFold 2) "sees" "pressure" energy near exits and automatically distributes pedestrians evenly, like perfectly folding protein. Practical implementation, visualization, and results analysis.

#### Self-Study Materials

* **[study_materials.md](./en/study_materials.md)**
    * **Topic:** Materials for study and self-learning plan.
    * **Content:** Structured materials for academic study and research, including simplified versions of key modules (classical algorithm challenge, AlphaFold 2 biological breakthrough, TSP-PFP mathematical isomorphism, new GDL paradigm). Self-study plan with two modules: "Classical crisis and biological bridge" and "From formulas to simulation". Case for presentation "Stadium evacuation" comparing classical approach (A*) and GDL. Topics for course/thesis projects. Academic analogy about difference between classical algorithms and AlphaFold 2-based approach.
