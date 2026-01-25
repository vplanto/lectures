# Study Materials and Self-Learning Plan

> 🇺🇦 [Українська версія](../study_materials.md) | 🇬🇧 English version

## 1. Study Materials (Theoretical Block)

Following modules presented in simplified format for academic study and research:

• Real-world challenge: Why classical algorithms (A*, Branch & Bound) "die" when trying to plan routes for hundreds of couriers on Black Friday due to curse of dimensionality

**Details:** [02_algorithm_limitations.md](./02_algorithm_limitations.md) — classical algorithm limitations and curse of dimensionality

• Biological breakthrough: How AlphaFold 2 model solved Levinthal's paradox problem (protein conformation enumeration that would take longer than universe's age)

**Details:** 
- [03_protein_folding_problem.md](./03_protein_folding_problem.md) — Levinthal's paradox and energy landscape
- [04_alphafold2_architecture.md](./04_alphafold2_architecture.md) — AlphaFold 2 architecture

• Mathematical isomorphism: Proof that Traveling Salesman Problem (TSP) is identical to Protein Folding Problem (PFP)

**Amino acid is client, and free energy is delivery cost**

**Details:** [05_np_completeness_isomorphism.md](./05_np_completeness_isomorphism.md) — mathematical problem isomorphism

• New paradigm (GDL): Instead of blindly enumerating variants, we teach neural network to "feel" space geometry and be invariant to map rotations (Geometric Deep Learning)

**Details:** 
- [00_geometric_deep_learning_intro.md](./00_geometric_deep_learning_intro.md) — Geometric Deep Learning introduction
- [06_tsp_via_af2_geometry.md](./06_tsp_via_af2_geometry.md) — GDL adaptation for logistics

---

## 2. Self-Study Plan

### Module #1: "Classical Crisis and Biological Bridge"

• Discussion why O(2^N) is death sentence for large city logistics

**Materials:** [02_algorithm_limitations.md](./02_algorithm_limitations.md)

• Explaining "atom as agent" idea: how local atom interactions create complex protein structure, and how to apply this to couriers

**Materials:** 
- [03_protein_folding_problem.md](./03_protein_folding_problem.md)
- [06_tsp_via_af2_geometry.md](./06_tsp_via_af2_geometry.md) — agent interpretation section
- [12_emergent_behavior.md](./12_emergent_behavior.md) — emergent behavior

### Module #2: "From Formulas to Simulation"

• Recycling mechanism breakdown: how iterative structure refinement acts as step-by-step motion simulation in time

**Materials:** 
- [04_alphafold2_architecture.md](./04_alphafold2_architecture.md) — Recycling mechanism
- [06_tsp_via_af2_geometry.md](./06_tsp_via_af2_geometry.md) — dynamic flow simulation

• Preparing for final case: comparing "individual" pathfinding (A*) and "collective" intelligence (GDL)

**Materials:** [13_evacuation_workshop.md](./13_evacuation_workshop.md) — evacuation workshop

---

## 3. Presentation Case: "Stadium Evacuation"

Case goal — analyze ready example comparing two approaches for evacuating 10,000 people

**Detailed workshop:** [13_evacuation_workshop.md](./13_evacuation_workshop.md)

### Theoretical Part

Energy landscape as key concept. In logistics "high energy" means congestion and conflicts of interest

**Materials:** 
- [03_protein_folding_problem.md](./03_protein_folding_problem.md) — energy landscape
- [12_emergent_behavior.md](./12_emergent_behavior.md) — congestion analysis as mathematical obstacles

### Classical Approach (A*)

Each pedestrian goes to nearest exit. Result — catastrophic congestion, because algorithm doesn't "see" other people. **Evacuation time — 60 min**

**Materials:** [02_algorithm_limitations.md](./02_algorithm_limitations.md) — A* limitations

### GDL Example (Neural Network Geometry)

Model works like AlphaFold 2. It senses "pressure" near exits and automatically redistributes people evenly. Result — no congestion. **Evacuation time — 50 min**

**Materials:** 
- [04_alphafold2_architecture.md](./04_alphafold2_architecture.md) — AlphaFold 2 architecture
- [06_tsp_via_af2_geometry.md](./06_tsp_via_af2_geometry.md) — logistics adaptation

---

## 4. Topics for Course/Thesis Projects (Future)

Based on these materials, directions for academic research can be chosen:

1. **"Evoformer Architecture Adaptation for VRPTW Problems"** (with time windows as 4D-corridors)

   **Base materials:** 
   - [04_alphafold2_architecture.md](./04_alphafold2_architecture.md) — Evoformer
   - [06_tsp_via_af2_geometry.md](./06_tsp_via_af2_geometry.md) — 4D space geometry
   - [07_implementation_methodology.md](./07_implementation_methodology.md) — implementation methodology

2. **"Traffic Jam Prediction as Energy Landscape Bottlenecks"**

   **Base materials:**
   - [03_protein_folding_problem.md](./03_protein_folding_problem.md) — energy landscape
   - [12_emergent_behavior.md](./12_emergent_behavior.md) — congestion analysis
   - [06_tsp_via_af2_geometry.md](./06_tsp_via_af2_geometry.md) — bottleneck detection

3. **"Agent Flow Simulation with 10,000 RPS Throughput Based on GPU"**

   **Base materials:**
   - [12_emergent_behavior.md](./12_emergent_behavior.md) — agent simulation
   - [06_tsp_via_af2_geometry.md](./06_tsp_via_af2_geometry.md) — dynamic flow simulation
   - [07_implementation_methodology.md](./07_implementation_methodology.md) — technical implementation

---

## Academic Analogy

**Classical A* algorithm** is like pedestrian walking to exit with flashlight, seeing only meter ahead.

**AlphaFold 2-based approach** is like turning on lights across entire stadium simultaneously: everyone sees everyone and entire system instantly "settles" into optimal state, like protein folding perfectly into living cell.
