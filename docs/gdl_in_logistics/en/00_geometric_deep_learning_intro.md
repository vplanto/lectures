# 00_geometric_deep_learning_intro.md: Geometric Deep Learning in Logistics

> 🇺🇦 [Українська версія](../00_geometric_deep_learning_intro.md) | 🇬🇧 English version

**Course:** Geometric Deep Learning in Logistics
**Module:** 0. Course Introduction
**Level:** Advanced / Expert

---

## 1. Scaling Problem: Why Classical Algorithms Die

### 1.1. Real Trigger: Black Friday Routing System Crash

In 2021, one of Ukraine's largest e-commerce platforms faced catastrophe: the delivery planning system stopped responding during peak load. Instead of 5-10 seconds to generate routes for 200 couriers, the system hung for 2-3 minutes or crashed entirely with `OutOfMemoryError`.

**Technical breakdown:**
- **Classical approach:** Branch & Bound (Gurobi Optimizer) on CPU
- **Problem size:** 200 clients, 50 vehicles, 3D coordinates + time windows
- **Complexity:** $O(2^{200})$ worst case
- **Memory:** Distance matrix $200 \times 200 = 40,000$ elements × 8 bytes = 320 KB (not a problem)
- **Real problem:** Branch & Bound search tree creates $10^9+$ nodes, each requiring ~1 KB for state storage (bounds, constraints). This is $>1$ TB RAM.

**Physical limitation:**
Even on server with 512 GB RAM, operating system cannot allocate more than 256 GB to single process (64-bit address space limitation). Algorithm dies not from mathematics, but from memory architecture.

### 1.2. Curse of Dimensionality

Classical optimization algorithms (simplex method, B&B) work in **Euclidean space** with fixed dimensionality. But logistics problem isn't just coordinates $(x, y)$.

**Full state space for VRP:**
- Geographic coordinates: 2D or 3D
- Arrival time: 1D
- Remaining fuel/charge: 1D
- Cargo loading: 1D
- Visited nodes history: $N$ bits (one-hot encoding)

**Total dimensionality:** $d = 2 + 1 + 1 + 1 + N = N + 5$

For $N=200$ clients this is $d=205$ dimensional space. Number of points needed for uniform hypercube $[0,1]^{205}$ coverage equals $k^{205}$, where $k$ is points per axis. Even for $k=2$ this is $2^{205} \approx 10^{61}$ points.

**Conclusion:** No discrete algorithm can enumerate this space. Need **continuous** approach that "feels" space structure, not enumerates it.

---

## 2. Geometric Deep Learning: Mathematical Foundation

### 2.1. Definition Through Group Theory

**Geometric Deep Learning (GDL)** is a class of neural network architectures that preserve **invariance** with respect to space symmetry transformations.

**Formal definition:**
Let $G$ be transformation group (e.g., rotations, reflections, translations). Function $f: X \to Y$ is called **$G$-invariant** if:
$$f(g \cdot x) = f(x), \quad \forall g \in G, x \in X$$

**Example for logistics:**
If we rotate entire city map around depot by $90°$, optimal route must remain optimal (up to coordinate rotation). Classical neural network (MLP) doesn't guarantee this — it learns specific orientation. GDL architecture is automatically invariant.

### 2.2. Graph as Geometric Object

**Key idea:** Cities and roads aren't just points and lines. It's a **graph** with geometric structure.

**Mathematical model:**
Graph $G = (V, E)$ where:
- $V = \{v_1, \dots, v_N\}$ — vertex set (cities, clients)
- $E \subseteq V \times V$ — edge set (roads, possible movements)
- Each vertex has **features** $h_i \in \mathbb{R}^d$ (coordinates, demand, time windows)
- Each edge has **edge features** $e_{ij} \in \mathbb{R}^k$ (distance, travel time, cost)

**Geometric structure:**
Distance between vertices $v_i$ and $v_j$ is defined not only by Euclidean metric $\|x_i - x_j\|_2$, but also **graph metric** (shortest path distance):
$$d_G(v_i, v_j) = \min_{p \in \text{paths}(i,j)} \sum_{(u,v) \in p} w_{uv}$$

Where $w_{uv}$ is edge weight (time, cost, distance).

### 2.3. Message Passing: How Neural Network "Sees" Graph

**Classical neural network (MLP)** takes fixed vector $x \in \mathbb{R}^n$ as input. But graph has **variable structure** — each vertex has different number of neighbors.

**Solution: Message Passing Neural Network (MPNN):**

At each layer $l$ vertex $v_i$ receives "messages" from neighbors and updates its representation:

$$
h_i^{(l+1)} = \text{UPDATE}^{(l)}\left(h_i^{(l)}, \text{AGGREGATE}^{(l)}\left( \{h_j^{(l)} : j \in \mathcal{N}(i)\} \right)\right)
$$

Where:
- $\mathcal{N}(i)$ — neighbor set of vertex $i$
- $\text{AGGREGATE}$ — aggregation function (sum, mean, maximum)
- $\text{UPDATE}$ — neural network (MLP) that updates features

**Why this works:**
After $L$ layers, vertex $v_i$ "sees" all vertices at distance $\le L$ (receptive field). For routing problem this means model accounts for not only direct neighbors, but **global structure** of graph.

**Complexity:**
- One layer: $O(\lvert E \rvert \cdot d)$ operations (where $d$ is feature dimension)
- $L$ layers: $O(L \cdot \lvert E \rvert \cdot d)$
- For sparse graph (each city connected to $k \ll N$ neighbors): $\lvert E \rvert = O(kN)$, so total complexity $O(L \cdot k \cdot N \cdot d)$ — **linear** in number of vertices!

Compare with Branch & Bound: $O(2^N)$.

---

## 3. AlphaFold 2: Biological Analogy as Engineering Solution

### 3.1. Levinthal's Paradox and Energy Landscape

In 1969, Cyrus Levinthal formulated paradox: protein with $N=100$ amino acids has $\approx 3^{100}$ possible conformations (shapes). Even if each conformation forms in $10^{-13}$ seconds, enumerating all variants takes $10^{27}$ years — more than universe's age. But proteins fold in milliseconds.

**Conclusion:** Proteins don't enumerate conformations. They "feel" energy landscape and descend to global minimum of free energy $E(\mathbf{r}_1, \dots, \mathbf{r}_N)$.

**Mathematical analogy with VRP:**
- **Protein:** Minimize $E(\mathbf{r}_1, \dots, \mathbf{r}_N)$ over amino acid coordinates
- **VRP:** Minimize $C(\pi_1, \dots, \pi_K)$ over visit sequences (permutations)

Both problems are finding global minimum on **high-dimensional landscape** with exponential number of local minima.

### 3.2. AlphaFold 2 Architecture: Evoformer and Geometry Tower

**AlphaFold 2** (DeepMind, 2020) solved protein folding problem with accuracy competing with experimental methods. Key idea — division into two stages:

1. **Evoformer:** Processes **evolutionary information** (Multiple Sequence Alignment, MSA) and builds pairwise representation (interaction matrix between amino acid pairs)
2. **Geometry Tower:** Converts pairwise representation into **3D coordinates** of atoms

**Mapping to VRP:**

| AlphaFold 2 | VRP / TSP |
|-------------|-----------|
| Amino acid | Client / city |
| Polypeptide chain | Vehicle route |
| MSA (evolutionary history) | Traffic history / delivery patterns |
| Pairwise representation | Distance matrix / interactions |
| 3D structure | Optimal visit sequence |
| Free energy | Total route cost |

**Why this works:**
Evoformer uses **Attention mechanism** to detect long-range dependencies. If two amino acids frequently co-occur across species, they're probably close in 3D space. Similarly: if two clients are often delivered together (pattern in data), they should probably be in same route.

### 3.3. Invariant Point Attention (IPA): Geometric Invariance

**Problem:** Classical Attention (Transformer) is not rotation-invariant. If we rotate coordinates, result changes.

**AlphaFold 2 Solution: Invariant Point Attention**

Instead of processing coordinates directly, IPA works with **relative positions** and **angles**:

$$
\text{IPA}(\mathbf{q}, \mathbf{k}, \mathbf{v}, \mathbf{T}) = \text{Softmax}\left( \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}} + \text{GeomBias}(\mathbf{T}) \right) \mathbf{v}
$$

Where $\mathbf{T}$ is transformation tensor (rotations + translations), and $\text{GeomBias}$ is function depending only on **relative** positions, not absolute.

**For logistics:**
We don't want model to depend on specific coordinate system (e.g., whether using WGS84 or UTM). IPA automatically makes model invariant to rotations and translations.

---

## 4. Problem Isomorphism: Proof Through Reduction

### 4.1. Formal NP-Completeness Proof

**Theorem:** Protein Folding Problem (PFP) and Traveling Salesman Problem (TSP) are **NP-equivalent**.

**Proof (direction TSP → PFP):**

Let TSP instance have $N$ cities and distance matrix $D_{ij}$. Construct protein:

1. Create $N$ amino acids arranged in linear chain
2. Set interaction energy between amino acids $i$ and $j$:
   $$E_{ij} = -D_{ij}$$
   (minus because we minimize energy, and in TSP minimize distance)

3. Add constraint: amino acids can fold only to form cycle (closed tour)

4. Global energy minimum $E = -\sum_{(i,j) \in \text{tour}} D_{ij}$ corresponds to optimal TSP tour.

**Proof (direction PFP → TSP):**

Let protein have $N$ amino acids and energy function $E(\mathbf{r}_1, \dots, \mathbf{r}_N)$. Construct TSP:

1. Each amino acid becomes city
2. Distance between cities $i$ and $j$:
   $$D_{ij} = \min_{\mathbf{r}_1, \dots, \mathbf{r}_N} E(\mathbf{r}_1, \dots, \mathbf{r}_N) \text{ subject to } \|\mathbf{r}_i - \mathbf{r}_j\| = d_{target}$$

3. Optimal TSP tour corresponds to protein folding sequence.

**Conclusion:** If AlphaFold 2 efficiently solves PFP, its architecture can be adapted for TSP/VRP.

### 4.2. Why Geometric Approach Beats Combinatorial

**Combinatorial approach (Branch & Bound):**
- Enumerates **discrete** states (permutations)
- Each state is separate node in search tree
- Complexity: $O(2^N)$ nodes

**Geometric approach (AlphaFold 2 / GDL):**
- Works with **continuous** space (coordinates, embeddings)
- Gradient descent finds local minimum in $O(N^2)$ operations
- Attention mechanism allows "jumping" through local minima barriers

**Physics analogy:**
Combinatorial approach is like enumerating all possible atom configurations in crystal. Geometric approach is like **molecular dynamics**: system itself finds energy minimum through continuous evolution.

---

## 5. Engineering Implementation: Memory, Parallelism, Throughput

### 5.1. Memory Layout for Graph Neural Networks

**Problem:** Graph has variable structure. How to efficiently store it for GPU?

**Solution: Batching with padding or CSR format**

**Variant A: Dense Batching (simple but inefficient)**
```python
# Bad: for graph with N=10 and N=1000 vertices use same size
batch_size = 32
max_nodes = 1000  # Largest graph in batch
node_features = torch.zeros(batch_size, max_nodes, d)  # 32 × 1000 × 128 = 4.1 MB
adjacency = torch.zeros(batch_size, max_nodes, max_nodes)  # 32 × 1000 × 1000 = 128 MB
```
**Problem:** 99% of memory spent on padding (zeros). For graph with 10 nodes we allocate memory for 1000.

**Variant B: CSR Format (efficient but complex)**
```python
# Good: store only real edges
class GraphBatch:
    node_features: Tensor  # [total_nodes, d] - all nodes from all graphs
    edge_index: Tensor     # [2, total_edges] - edge indices
    batch_ptr: Tensor      # [batch_size+1] - pointers to each graph start
```
**Advantages:**
- Memory: $O(\sum_i \lvert V_i \rvert + \sum_i \lvert E_i \rvert)$ instead of $O(\text{batch\_size} \times \max_i \lvert V_i \rvert^2)$
- For batch with graph sizes [10, 50, 1000]: memory savings $>90\%$

**Conclusion:** Use **PyTorch Geometric** or **DGL** — they automatically convert graphs to CSR format.

### 5.2. GPU Parallelism: Sparse Matrix Multiplication

**Message Passing operation:**

$$
H^{(l+1)} = \sigma(A H^{(l)} W^{(l)})
$$

Where $A$ is adjacency matrix (sparse), $H^{(l)}$ is node features, $W^{(l)}$ is layer weights.

**CPU complexity:**
- Dense multiplication: $O(N^2 \cdot d)$
- Sparse (CSR): $O(\lvert E \rvert \cdot d)$

**On GPU (cuSPARSE):**
- Dense: $O(N^2 \cdot d / \text{cores})$ — limited by memory bandwidth
- Sparse: $O(\lvert E \rvert \cdot d / \text{cores})$ — more efficient, reads only non-zero elements

**Throughput for VRP with $N=200$:**
- CPU (Gurobi B&B): 1 instance per 60 seconds
- GPU (GDL model, batch\_size=32): 32 instances per 0.1 seconds
- **Speedup: $>19,000\times$**

### 5.3. Latency vs Throughput: Online vs Batch Inference

**Online mode (real-time):**
Client places order → system generates route in $<1$ second.

**Architecture:**
```
Request → GDL Model (single instance) → Route
Latency: ~50-100 ms (GPU inference)
```

**Batch mode (offline optimization):**
Accumulate orders for day → optimize all routes simultaneously.

**Architecture:**
```
Daily Orders → GDL Model (batch_size=1000) → All Routes
Throughput: 1000 routes in ~200 ms
Cost per route: 0.2 ms (500× cheaper)
```

**Conclusion:** For real-time use lightweight model (fewer layers, fewer parameters). For batch — full model with maximum accuracy.

---

## 6. Engineering Challenge: AI-Resistant Assessment

### 6.1. Task: Design Routing System for 10,000 RPS

**Context:**
You're building delivery system for megacity. Peak load: 10,000 requests per second (RPS). Each request is VRP problem with $N=50-200$ clients.

**Requirements:**
- Latency: $P_{99} < 500$ ms (99% requests processed faster than 500 ms)
- Throughput: sustain 10,000 RPS
- Accuracy: routes must be within 5% of optimum (relative to manual calculation)

**Constraints:**
- Budget: maximum 10 GPU (NVIDIA A100, 80 GB)
- Network: latency between servers $<10$ ms
- Memory: each GPU has 80 GB, but model occupies 20 GB

**Your task:**

1. **Choose architecture:**
   - A) One large server with 10 GPU (single node)
   - B) Distributed system: 5 servers with 2 GPU each (multi-node)
   - C) Hybrid: 1 GPU for real-time (lightweight model), 9 GPU for batch (full model)

2. **Justify choice:**
   - Calculate maximum throughput for each variant
   - Estimate latency (accounting for network overhead for multi-node)
   - Calculate cost (GPU-hours)

3. **Defend solution:**
   - Why is your approach better than alternatives?
   - What trade-offs did you accept?
   - How does system scale if load grows to 50,000 RPS?

**Evaluation criteria:**
- **Insufficient:** "Use PyTorch and run on GPU" (no architectural thinking)
- **Good:** Throughput/latency calculation with concrete numbers
- **Excellent:** Trade-off analysis, scalability, justification through metrics (not intuition)

### 6.2. Reference Solution (for instructor)

**Recommended architecture: C) Hybrid**

**Justification:**

**Real-time path (1 GPU):**
- Model: 4 layer MPNN, $d=64$ (embedding dimension)
- Parameters: ~2M (occupies 8 MB memory)
- Throughput: 1000 instances/second on A100
- Latency: $P_{50}=30$ ms, $P_{99}=80$ ms

**Batch path (9 GPU):**
- Model: 12 layer MPNN + Attention, $d=256$
- Parameters: ~50M (occupies 200 MB memory)
- Batch size: 1000 instances per GPU
- Throughput: 9 × 100 instances/second = 900 instances/second
- Latency: 1-2 seconds (acceptable for offline optimization)

**Load distribution:**
- 90% requests → real-time (1 GPU): 9,000 RPS
- 10% requests → batch (9 GPU): 1,000 RPS → processed in 1.1 seconds

**Scaling:**
If load grows to 50,000 RPS:
- Add 4 GPU for real-time (now 5 GPU, each processes 10,000 RPS)
- Batch remains unchanged (10% = 5,000 RPS, processed in 5.5 seconds)

**Trade-offs:**
- ✅ Low latency for most requests
- ✅ High accuracy for complex cases (batch)
- ❌ Complexity: need queue system for request distribution
- ❌ Additional overhead: load balancer, monitoring

---

## 7. Sources and Literature

### 7.1. Fundamental Geometric Deep Learning Theory
* **Book:** *Bronstein, M. M., et al. "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges".* [arXiv:2104.13478](https://arxiv.org/abs/2104.13478) — GDL mathematical foundation, group theory, invariance.
* **Paper:** *Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks".* [ICLR 2017](https://arxiv.org/abs/1609.02907) — Classical GCN architecture, message passing.
* **Paper:** *Vaswani, A., et al. (2017). "Attention is All You Need".* [NeurIPS 2017](https://arxiv.org/abs/1706.03762) — Transformer architecture, foundation for Evoformer.

### 7.2. AlphaFold 2 and Biological Analogy
* **Paper:** *Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold".* [Nature 2021](https://www.nature.com/articles/s41586-021-03819-2) — Original AlphaFold 2 publication, detailed architecture description.
* **Paper:** *Baek, M., et al. (2021). "Accurate prediction of protein structures and interactions using a three-track neural network".* [Science 2021](https://www.science.org/doi/10.1126/science.abj8754) — AlphaFold 2 competitor (RoseTTAFold), alternative approach.
* **Video:** [DeepMind: AlphaFold 2 Technical Talk](https://www.youtube.com/watch?v=GGjfXc3hW2A) — Technical architecture presentation from developers.

### 7.3. GDL Applications in Logistics and Optimization
* **Paper:** *Kool, W., et al. (2019). "Attention, Learn to Solve Routing Problems!".* [ICLR 2019](https://arxiv.org/abs/1803.08475) — Transformer for TSP/VRP, first attempt at applying attention to combinatorial optimization.
* **Paper:** *Bresson, X., & Laurent, T. (2021). "The Transformer Network for the Traveling Salesman Problem".* [arXiv:2103.03012](https://arxiv.org/abs/2103.03012) — Improved version, comparison with classical methods.
* **Resource:** [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — Library for working with graphs and GNN, GPU-optimized.

### 7.4. Engineering Practice: Scaling and Performance
* **Book:** *Dean, J., & Ghemawat, S. (2008). "MapReduce: Simplified Data Processing on Large Clusters".* [OSDI 2004](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf) — Fundamental principles of distributed systems (relevant for batch processing).
* **Paper:** *Wang, M., et al. (2019). "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks".* [arXiv:1909.01315](https://arxiv.org/abs/1909.01315) — DGL, PyTorch Geometric alternative, optimizations for large graphs.
* **Resource:** [NVIDIA cuSPARSE Documentation](https://docs.nvidia.com/cuda/cusparse/index.html) — GPU sparse matrix operations optimization.

---

**Next step:** Transition to detailed breakdown of classical transport problem and its limitations ([01_classical_transport_problem.md](./01_classical_transport_problem.md)).
