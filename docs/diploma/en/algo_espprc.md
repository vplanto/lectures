# Shortest Path Problem with Resource Constraints (ESPPRC)

> 🇺🇦 [Українська версія](../algo_espprc.md) | 🇬🇧 English version

## Problem Statement

Let $G=\left(V,A\right)$ be a graph, where $A$ is the set of arcs and $V = \{v_1, \dots, v_n\}$ is the set of nodes including the origin node (from which we depart) $p$ and the destination node $d$ (which must be reached).

A cost $c_{\text{ij}}$ is associated with each arc $\left(v_i,v_j\right){\ in}A$.

Let $L$ be the number of resources and $d_{\text{ij}^l}\ge 0$ be the consumption of resource $l$ along arc $\left(v_i,v_j\right)$. With each node $v_i$ and each resource $l$, two non-negative values $a_{i^l}$ and $b_{i^l}$ are associated, such that resource $l$ consumption on route from $p$ to $v_i$ is bounded by interval $\left[a_{i^l},b_{i^l}\right]$. If resource $l$ consumption is below $a_{i^l}$ when route reaches $v_i$, consumption is set to $a_{i^l}$.

Note that this is natural for time resource, but also allows us to introduce capacity constraints by defining intervals $\left[0,Q\right]$ at nodes, where $Q$ is capacity bound.

**The goal** is to obtain minimum-cost elementary route from $p$ to $d$ that satisfies all resource constraints.

---

## Label Correcting Algorithm for Solving ESPPRC

### Route Label Definition

**Definition XX.2.1:** With each route $X_{\text{pi}}$ from origin node $p$ to node $v_i$, an associated state $R_i=\left(T_{i^1},T_{i^2},\ldots ,T_{i^L},s_i,V_{i^1},\ldots ,V_{i^n}\right)$ corresponds to the amount of each resource used by the route, number of visited nodes, and visitation vector ($V_{i^k}=1$ if route visits node $v_k$, 0 otherwise).

**Definition XX.2.2:** Let $X_{\text{pi}}^{'}$ and $X^{\ast _{\text{pi}}}$ be two different routes from $p$ to $v_i$ with associated labels $\left(R_i^{'},C_i^{'}\ right)$, $\left(R_{i^{\ast }},C_{i^{\ast }}\right)$. $X_{\text{pi}}^{'}$ **dominates** $X^{\ast _{\text{pi}}}$ if and only if $C_i^{'}\le C^{\ast _i}$, $s_i^{'}\le s^{\ast _i}$, $T_{i^l}^{'}\le T_{i^{\ast l}}$ for $l=1,\ldots ,L$, $V^{k_i}\le V^{\ast k_i}$ for $k=1,\ldots ,n$, and $\left(R_i^{'},C_i^{'}\right)\neq \left(R_{i^{\ast }},C_{i^{\ast }}\right)$.

Note that $s_i=\sum _{K=1}^nV_{i^K}$ and that this resource is needed only for computational purposes, since a label cannot dominate another label if it visited more nodes.

The algorithm principle is as follows: all non-dominated routes are extensions of non-dominated routes and, to obtain optimal problem solution, only non-dominated routes need to be considered (Desrochers [7]). Indeed, extending dominated route $X_{\text{pi}}$ with arc $(v_i,v_j)$ results in route that is either dominated or equal to extension $X_{\text{pj}}^{'}$ of route $X_{\text{pi}}^{'}$ which dominates route $X_{\text{pi}}$.

This label definition guarantees that we generate only labels corresponding to elementary paths, and we obtain optimal problem solution. However, state space size increases significantly and many labels must be generated and stored during solution finding.

### Label Definition Improvement

Consider another label definition to limit their number during solving. Let us consider partial route and its associated label. In previous definition, nodes belonging to route were stored in label's visitation vector. Since these nodes were already visited, they cannot be visited in any extension of corresponding route. Moreover, there may be other nodes that cannot be visited in any extension of corresponding route due to resource constraints.

The new definition principle is to more effectively determine which nodes cannot be visited regardless of reason. Let's call such nodes **unreachable**. When label is extended, visitation resources corresponding to nodes that cannot be reached (either because they were already visited, or due to resource constraint) will be used.

This algorithmic modification is computationally attractive, as dominance relation becomes more prominent, as shown in following example.

#### Example (Dominance)

Consider ESPPRC example: given two nodes $v_1$, $v_2$ and two labels $\lambda _2^a=\left(10,s_2^a,0,1,C_2^a\right)$ and $\lambda _2^b=\left(5,s_2^b,1,1,C_2^b\right)$ at node $v_2$. Resources correspond to time, number of visited nodes, and visitation vector restricted to $\{v_1,v_2\}$. Assume that $s_2^b\le s_2^a$, $C_2^b\le C_2^a$ and that $V_2^{\text{kb}}\le V_2^{\text{ka}}$ for all other graph nodes. Also assume that label $\lambda _2^a$ cannot be extended to node $v_1$ due to time resource constraint.

With dominance relation from previous section, neither of these two labels dominates the other. However, one can confidently say that any extension of $\lambda _2^a$ to destination node can be copied for label $\lambda _2^b$, and that route resulting from label $\lambda _2^a$ extension will be more expensive. Therefore, using unreachability concept, labels become $\lambda _2^a=\left(10,s_2^a,1,1,C_2^a\right)$ and $\lambda _2^b=\left(5,s_2^b,1,1,C_2^b\right)$. It follows that label $\lambda _2^b$ dominates label $\lambda _2^a$.

#### Unreachability and Non-Dominance Solution Lemma

**Definition XX.2.3:** For each route $X_{\text{pi}}$ from origin node $p$ to node $v_i{\in}V$, node $v_k$ is said to be **unreachable** if it is already included in $X_{\text{pi}}$ or if there exists resource $l{\in}\left\{1,...,L\right\}$ satisfying $T_i^l+d_{\text{ik}}^l>b_k^l$ (meaning current consumption value prevents reaching node $v_k$).

**Definition XX.2.4:** With each path $X_{\text{pi}}$ from origin node $p$ to node $v_i{\in}V$, an associated state $R_i=\left(T_{i^1},T_{i^2},\ldots ,T_{i^L},s_i,V_{i^1},\ldots ,V_{i^n}\right)$ corresponds to amount of resources used by route, number of unreachable nodes, and unreachable node vector, defined as $V_i^K=1$ if node $v_k$ is unreachable.

Node is said to be unreachable when it cannot be reached directly using outgoing arc. This also means there is no route allowing this node to be reached, since triangle inequality has been introduced for resources. Note also that still $s_i=\sum _{k=1}^nV_i^K$.

With aforementioned definitions, we can preserve same dominance relation and only consider non-dominated routes.

**Lemma (on solution non-dominance):** During modified algorithm execution, we must only consider non-dominated routes.

Given label modification algorithm is easy to implement. Indeed, when expanding label, it is only necessary to recalculate unreachable node set. For this, we evaluate feasibility of label extension through each outgoing arc. Note also that computation time depends on number of resources $L$. When $L$ is too large, unreachable node definition can be adapted to limit computation time; one can only recalculate unreachable nodes for some resource subset. Otherwise, when $L$ is small, only slightly more time is needed to determine unreachable nodes than to store nodes that were visited. Full algorithm description is given below.

---

## Algorithm Pseudocode

Introduce following notation:

* $\Lambda _i$: Label list at node $v_i$.
* $\text{Succ}\left(v_i\right)$: Set of nodes associated with node $v_i$.
* $E$: List of nodes to consider.
* $\text{Extend}\left(\lambda _i,v_j\right)$: Function that returns label resulting from extension from label $\lambda _i{\in}\Lambda $ to node $v_j$, when extension is possible, null value otherwise. Function first updates resource consumption $l=1,\ldots ,L$. If resource constraints are satisfied, it explores outgoing arc set to update unreachable node vector and number of unreachable nodes.
* $F_{\text{ij}}$: Set of labels that are extensions from node $v_i$ to node $v_j$.
* $\text{EFF}\left(\Lambda \right)$: Procedure that keeps only non-dominated labels in label list $\Lambda $.

Now describe procedure $\text{Espprc}(p)$ which determines all non-dominated routes starting at node $p$ to any graph nodes.

```pseudocode
Espprc(p)
 1  Initialization
 2  Λ_p ← {(0,...,0)}
 3  for all v_i ∈ V\{p}
 4    Λ_i ← ∅
 5  E = {p}
 6
 7  repeat
 8    Exploring node successors
 9    Select v_i ∈ E
10    for all v_j ∈ Succ(v_i)
11      Fi_j ← ∅
12      for all λ_i = (T_i^1,...,T_i^L, s_i, V_i^1,...,V_i^L, C_i) ∈ Λ_i
13        if V_i^j = 0
14          then F_ij ← F_ij ∪ {Extend(λ_i, v_j)}
15      Λ_j ← EFF(F_ij ∪ Λ_j)
16      if Λ_j changed
17        then E ← E ∪ {v_j}
18    Reducing E
19    E ← E\{v_i}
20  exit if E = ∅
```

This algorithm's execution time is directly related to graph structure, number of nodes, and resource constraint density. If problem is highly constrained, quite large problems can be solved.

-----

## Example (Network)

Consider following network with given arc travel costs (number above arc), resource consumption (shown in square brackets separated by commas for each resource type), and resource consumption windows (assumed same for all resources: 4 for first resource and 6 for second):

![ESPPRC Network Example](attachment/Espprc-img001.png)

Need to determine optimal elementary path from depot (node 1) to node 5. Procedure $\text{Espprc}$ produced following two pareto-optimal solutions:

$$
\Delta_1 = \left[4,5,4,1,1,0,1,1,11\right]
$$
$$
\Delta_2 = \left[4,6,4,1,1,1,0,1,9\right]
$$

Here numbers in square brackets mean: first resource consumption, second resource consumption, number of visited nodes, node visitation vector (1 if node visited, 0 otherwise), and travel cost from depot to destination and back (back without resource accounting). As we see, following situation emerged: route 1 consumes less resource 2 but is more expensive in travel cost, while route 2 is more expensive in resource 2 but cheaper in travel cost.

If we introduce resource consumption windows of 4 for first resource and 5 for second, then answer will be only one:

$$
\Delta_1 = \left[4,5,4,1,1,0,1,1,11\right]
$$

and route 2 will be rejected since it cannot reach destination due to resource 2 constraint.
