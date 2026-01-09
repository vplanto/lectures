# 08_ai_assisted_development.md: Промпт-інжиніринг для AI-асистентів

**Курс:** Геометричний Deep Learning в Логістиці
**Модуль:** 4. Практикум (Workshop)
**Рівень:** Advanced / Expert

---

## 1. Реальний тригер: Чому LLM генерує математично неправильний код

### 1.1. Production Case: Помилка в Triangular Multiplicative Update (2024)

У 2024 році розробник намагався використати Claude для імплементації Triangular Multiplicative Update з модуля 07. Запит був простим:

```
"Implement triangular multiplicative update for pairwise representation in PyTorch"
```

**Результат (неправильно):**
```python
def triangular_update(P, G):
    N = P.shape[0]
    P_out = torch.zeros_like(P)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                P_out[i, j] += P[i, k] * G[k, j]  # Broadcasting error!
    return P_out
```

**Проблема:**
- `P[i, k]` має розмір `[c]` (остання розмірність)
- `G[k, j]` має розмір `[c]`
- `P[i, k] * G[k, j]` — це element-wise multiplication `[c] * [c] = [c]`
- Але якщо `P` має розмір `[N, N, c]`, то `P[i, k]` — це `[c]`, а не `[N, c]`

**Реальна помилка:**
Код не компілюється, бо PyTorch намагається broadcast `[c]` до `[N, c]`, що неможливо.

**Чому це сталося:**
LLM не отримав достатньо контексту про:
1. Точну структуру тензорів (розмірності)
2. Математичну формалізацію операції
3. Обмеження пам'яті (не можна використовувати цикли)

**Висновок:** Наївні промпти генерують наївний код. Потрібна структурована методологія промпт-інжинірингу.

---

## 2. Структура ефективного промпту: Математика → Код → Оптимізація

### 2.1. Трикомпонентна модель промпту

**Ефективний промпт складається з трьох частин:**

1. **Контекст (Context):** Математична формалізація, обмеження, вимоги
2. **Специфікація (Specification):** Точні розмірності тензорів, формат вхідних/вихідних даних
3. **Обмеження (Constraints):** Пам'ять, швидкість, сумісність

**Формула ефективного промпту:**
```
[ROLE] + [MATHEMATICS] + [TENSOR_SPEC] + [CONSTRAINTS] + [EXAMPLE]
```

### 2.2. Роль (Role): Встановлення контексту експерта

**Погано:**
```
"Write a function for triangular update"
```

**Добре:**
```
"You are an expert PyTorch developer specializing in geometric deep learning. 
You implement tensor operations with attention to memory efficiency and 
mathematical correctness."
```

**Чому це працює:**
- Встановлює експертний рівень
- Фокусує на конкретній області (geometric deep learning)
- Вказує на пріоритети (memory efficiency, correctness)

### 2.3. Математика (Mathematics): Формальна специфікація

**Погано:**
```
"Update pairwise representation using triangular multiplication"
```

**Добре:**
```
"Implement Triangular Multiplicative Update as defined in AlphaFold 2:

Given:
- P ∈ ℝ^(N×N×c): pairwise representation
- G ∈ ℝ^(N×N×c): gate tensor (learnable)

Compute:
- Outgoing: P_ij^out = Σ_k P_ik ⊙ G_kj  (element-wise ⊙ for each channel)
- Incoming: P_ij^in = Σ_k G_ik ⊙ P_kj
- Output: P_new = LayerNorm(P + P_out + P_in)

Where ⊙ denotes element-wise multiplication (Hadamard product) along the channel dimension."
```

**Чому це працює:**
- Чітка математична нотація
- Визначення всіх операцій
- Уточнення розмірностей

### 2.4. Специфікація тензорів (Tensor Specification): Точні розмірності

**Погано:**
```
"P and G are tensors"
```

**Добре:**
```
"Input tensors:
- P: torch.Tensor, shape [batch_size, N, N, c_pair], dtype float32
- G: torch.Tensor, shape [batch_size, N, N, c_pair], dtype float32

Output tensor:
- P_new: torch.Tensor, shape [batch_size, N, N, c_pair], dtype float32

Memory constraints:
- Must NOT create intermediate tensors of size [batch_size, N, N, N, c_pair]
- Use einsum or matrix multiplication, NOT nested loops
- Preserve gradients (operations must be differentiable)"
```

**Чому це працює:**
- Точні розмірності (включаючи batch dimension)
- Явні обмеження пам'яті
- Вимоги до диференційованості

### 2.5. Обмеження (Constraints): Інженерні вимоги

**Приклад:**
```
"Performance requirements:
- Must use vectorized operations (no Python loops over N)
- Must be compatible with mixed precision (FP16)
- Must support gradient checkpointing (no in-place operations)

Code style:
- Use type hints
- Add docstring with mathematical notation
- Include inline comments for complex operations"
```

### 2.6. Повний приклад промпту

```
You are an expert PyTorch developer specializing in geometric deep learning 
and tensor operations. You implement mathematically correct, memory-efficient 
code with attention to GPU optimization.

Implement Triangular Multiplicative Update as defined in AlphaFold 2:

Mathematical definition:
Given P ∈ ℝ^(N×N×c) and G ∈ ℝ^(N×N×c), compute:
- P_ij^out = Σ_k P_ik ⊙ G_kj  (element-wise ⊙ along channel dimension)
- P_ij^in = Σ_k G_ik ⊙ P_kj
- P_new = LayerNorm(P + P_out + P_in)

Tensor specifications:
- Input: P [batch_size, N, N, c_pair], G [batch_size, N, N, c_pair]
- Output: P_new [batch_size, N, N, c_pair]
- dtype: float32

Constraints:
- NO intermediate tensors of size [batch_size, N, N, N, c_pair]
- Use einsum or batched matrix multiplication
- NO Python loops over N (must be vectorized)
- Preserve gradients (differentiable operations only)
- Compatible with mixed precision training

Code requirements:
- Type hints
- Docstring with LaTeX math notation
- Inline comments for einsum operations
```

**Результат (правильно):**
```python
def triangular_multiplicative_update(
    P: torch.Tensor, 
    G: torch.Tensor
) -> torch.Tensor:
    """
    Triangular Multiplicative Update for pairwise representation.
    
    Computes:
    - P_ij^out = Σ_k P_ik ⊙ G_kj
    - P_ij^in = Σ_k G_ik ⊙ P_kj
    - P_new = LayerNorm(P + P_out + P_in)
    
    Args:
        P: Pairwise representation [batch_size, N, N, c_pair]
        G: Gate tensor [batch_size, N, N, c_pair]
    
    Returns:
        Updated pairwise representation [batch_size, N, N, c_pair]
    """
    # Outgoing: P[i,j] = sum_k P[i,k] * G[k,j] for each channel
    # einsum 'bikd,bkjd->bijd': for each batch b and channel d,
    # compute matrix multiplication P[:,:,d] @ G[:,:,d].T
    P_out = torch.einsum('bikd,bkjd->bijd', P, G)
    
    # Incoming: P[i,j] = sum_k G[i,k] * P[k,j] for each channel
    # einsum 'bikd,bkjd->bijd': G[:,:,d] @ P[:,:,d]
    P_in = torch.einsum('bikd,bkjd->bijd', G, P)
    
    # Layer normalization
    P_new = F.layer_norm(P + P_out + P_in, (P.shape[-1],))
    
    return P_new
```

---

## 3. Специфічні патерни для тензорних операцій

### 3.1. Патерн 1: Batched Matrix Operations

**Проблема:** LLM часто генерує цикли замість batched operations.

**Поганий промпт:**
```
"Multiply matrices A and B for each item in batch"
```

**Добре:**
```
"Perform batched matrix multiplication:
- A: [batch_size, N, M]
- B: [batch_size, M, K]
- Output: [batch_size, N, K]

Use torch.bmm or torch.einsum. NO loops over batch_size."
```

**Результат:**
```python
# Правильно
C = torch.bmm(A, B)  # [batch_size, N, K]

# Або
C = torch.einsum('bnm,bmk->bnk', A, B)
```

### 3.2. Патерн 2: Gathering та Indexing

**Проблема:** LLM часто використовує неправильний синтаксис для advanced indexing.

**Поганий промпт:**
```
"Extract values from tensor using indices"
```

**Добре:**
```
"Gather values from distance matrix D using route indices:
- D: [batch_size, N, N] - distance matrix
- route: [batch_size, route_length] - sequence of city indices (int64)
- Output: [batch_size, route_length] - distances between consecutive cities

Use torch.gather with proper dimension specification.
For route [i, j, k], extract D[:, i, j] and D[:, j, k]."
```

**Результат:**
```python
def extract_route_distances(D: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
    """
    Extract distances along route from distance matrix.
    
    Args:
        D: [batch_size, N, N] - distance matrix
        route: [batch_size, route_length] - city indices
    
    Returns:
        [batch_size, route_length-1] - distances between consecutive cities
    """
    batch_size, route_length = route.shape
    batch_indices = torch.arange(batch_size)[:, None].expand(-1, route_length - 1)
    
    from_cities = route[:, :-1]  # [batch_size, route_length-1]
    to_cities = route[:, 1:]     # [batch_size, route_length-1]
    
    distances = D[batch_indices, from_cities, to_cities]
    return distances
```

### 3.3. Патерн 3: Masking для Attention

**Проблема:** LLM часто створює неправильні маски для attention.

**Поганий промпт:**
```
"Apply mask to attention scores"
```

**Добре:**
```
"Apply causal mask to attention scores for autoregressive generation:
- logits: [batch_size, seq_len, seq_len] - attention scores
- mask: [batch_size, seq_len, seq_len] - boolean mask (True = allowed, False = masked)

Masked positions should have logits = -inf before softmax.
Use torch.masked_fill with proper broadcasting."
```

**Результат:**
```python
def masked_attention(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply mask to attention logits.
    
    Args:
        logits: [batch_size, seq_len, seq_len] - attention scores
        mask: [batch_size, seq_len, seq_len] - boolean mask (True = allowed)
    
    Returns:
        [batch_size, seq_len, seq_len] - masked attention probabilities
    """
    # Set masked positions to -inf
    masked_logits = logits.masked_fill(~mask, float('-inf'))
    
    # Softmax
    attn_probs = F.softmax(masked_logits, dim=-1)
    
    return attn_probs
```

### 3.4. Патерн 4: Gradient Checkpointing

**Проблема:** LLM не завжди розуміє, де застосовувати gradient checkpointing.

**Поганий промпт:**
```
"Add gradient checkpointing"
```

**Добре:**
```
"Implement gradient checkpointing for Evoformer layers:
- Apply checkpointing to HistoryStack and DistanceStack (memory-intensive)
- Do NOT checkpoint RouteModule (lightweight)
- Use torch.utils.checkpoint.checkpoint with proper function signature
- Ensure inputs are tensors (not tuples) for compatibility"
```

**Результат:**
```python
from torch.utils.checkpoint import checkpoint

class EvoformerWithCheckpointing(nn.Module):
    def __init__(self, num_layers=24):
        super().__init__()
        self.history_stack = HistoryStack(num_layers)
        self.distance_stack = DistanceStack(num_layers)
        self.route_module = RouteModule()
        self.use_checkpoint = True
    
    def forward(self, H, D):
        # Checkpoint memory-intensive stacks
        if self.use_checkpoint and self.training:
            H, D = checkpoint(self.history_stack, H, D, use_reentrant=False)
            D = checkpoint(self.distance_stack, D, H, use_reentrant=False)
        else:
            H, D = self.history_stack(H, D)
            D = self.distance_stack(D, H)
        
        # No checkpointing for lightweight route module
        route = self.route_module(D, H)
        
        return route
```

---

## 4. Промпти для JAX: Відмінності від PyTorch

### 4.1. Функціональна парадигма

**Ключова відмінність:** JAX використовує функціональне програмування, немає стану (state).

**Поганий промпт (PyTorch-style):**
```
"Create a neural network layer with learnable parameters"
```

**Добре (JAX-style):**
```
"Implement a JAX module using Flax or Haiku:
- Use functional API (no nn.Module classes)
- Parameters are explicit (not stored in module)
- Use jax.jit for compilation
- Support vmap for batching

Example structure:
```python
def layer_fn(params, x):
    # Forward pass
    return output

# Initialize parameters
params = init_layer_fn(rng, input_shape)
```
```

### 4.2. JAX-специфічні операції

**Промпт для JAX einsum:**
```
"Implement triangular update in JAX:
- Use jnp.einsum (not torch.einsum)
- Ensure jit-compatible (no Python loops)
- Support vmap for batching
- Use jax.lax.scan if needed for sequential operations

Memory: JAX uses XLA compilation, intermediate tensors are optimized automatically.
But still avoid explicit N^3 tensors in Python code."
```

**Результат:**
```python
import jax.numpy as jnp
from jax import jit, vmap

@jit
def triangular_update_jax(P, G):
    """
    JAX implementation of Triangular Multiplicative Update.
    
    Args:
        P: [N, N, c] - pairwise representation
        G: [N, N, c] - gate tensor
    
    Returns:
        [N, N, c] - updated pairwise representation
    """
    # Outgoing: P[i,j] = sum_k P[i,k] * G[k,j]
    P_out = jnp.einsum('ikd,kjd->ijd', P, G)
    
    # Incoming: P[i,j] = sum_k G[i,k] * P[k,j]
    P_in = jnp.einsum('ikd,kjd->ijd', G, P)
    
    # Layer normalization (JAX style)
    P_sum = P + P_out + P_in
    mean = P_sum.mean(axis=-1, keepdims=True)
    var = ((P_sum - mean) ** 2).mean(axis=-1, keepdims=True)
    P_new = (P_sum - mean) / jnp.sqrt(var + 1e-5)
    
    return P_new

# Batched version using vmap
triangular_update_batched = vmap(triangular_update_jax, in_axes=(0, 0))
```

### 4.3. Обробка випадковості в JAX

**Проблема:** JAX використовує explicit random keys, не глобальний генератор.

**Поганий промпт:**
```
"Add random dropout"
```

**Добре:**
```
"Implement dropout in JAX:
- Use jax.random for random number generation
- Pass rng_key explicitly (not global state)
- Use jax.random.bernoulli for mask generation
- Support jit compilation

Function signature:
```python
def dropout_fn(rng_key, x, dropout_rate=0.1):
    # Generate mask using rng_key
    # Apply mask to x
    return output
```
```

**Результат:**
```python
from jax import random

def dropout(rng_key, x, dropout_rate=0.1, training=True):
    """
    Dropout layer for JAX.
    
    Args:
        rng_key: JAX PRNGKey
        x: Input tensor
        dropout_rate: Probability of dropping out
        training: Whether in training mode
    
    Returns:
        Output tensor with dropout applied
    """
    if not training:
        return x
    
    # Generate mask
    keep_prob = 1.0 - dropout_rate
    mask = random.bernoulli(rng_key, keep_prob, x.shape)
    
    # Apply mask and scale
    output = x * mask / keep_prob
    
    return output
```

---

## 5. Системні промпти для Cursor/Claude

### 5.1. Структура системного промпту

**Системний промпт** встановлює контекст для всієї сесії. Він має бути:
- Стислим (1-2 абзаци)
- Специфічним до домену
- З пріоритетами

**Приклад системного промпту для Cursor:**
```
You are an expert in Geometric Deep Learning and tensor operations. 
You specialize in implementing AlphaFold 2-inspired architectures for 
combinatorial optimization (TSP/VRP).

Your code must be:
1. Mathematically correct (match LaTeX formulas exactly)
2. Memory-efficient (no O(N^3) intermediate tensors)
3. GPU-optimized (vectorized, no Python loops)
4. Well-documented (LaTeX math in docstrings)

When implementing tensor operations:
- Always specify tensor shapes explicitly
- Use einsum/bmm instead of loops
- Add type hints and docstrings
- Consider gradient checkpointing for large models

If unsure about tensor dimensions, ask for clarification rather than guessing.
```

### 5.2. Контекстні промпти для конкретних завдань

**Промпт для рефакторингу:**
```
"Refactor the following function to be memory-efficient:

[PASTE CODE]

Requirements:
- Replace nested loops with vectorized operations
- Use einsum for tensor contractions
- Add type hints and docstring
- Ensure backward compatibility (same input/output signature)"
```

**Промпт для дебагу:**
```
"Debug the following tensor operation. The error is:
[PASTE ERROR]

Code:
[PASTE CODE]

Please:
1. Identify the dimension mismatch
2. Explain the mathematical operation being performed
3. Provide corrected code with comments"
```

**Промпт для оптимізації:**
```
"Optimize this function for GPU performance:

[PASTE CODE]

Current issues:
- Uses Python loops (should be vectorized)
- Creates large intermediate tensors
- Not compatible with mixed precision

Provide optimized version with:
- Vectorized operations
- Memory-efficient implementation
- FP16 compatibility"
```

---

## 6. Типові помилки та як їх уникнути

### 6.1. Помилка 1: Неправильні розмірності тензорів

**Симптом:** LLM генерує код з неявними broadcast операціями, що призводять до помилок.

**Причина:** Недостатня специфікація розмірностей.

**Рішення:**
```
"Explicitly specify ALL tensor dimensions in comments:
- Input: [batch_size, N, N, c] 
- Intermediate: [batch_size, N, c]
- Output: [batch_size, N, N, c]

NO implicit broadcasting. All operations must have explicit dimension matching."
```

### 6.2. Помилка 2: Використання циклів замість векторізації

**Симптом:** LLM генерує `for` loops навіть коли це можна зробити векторізовано.

**Причина:** Недостатньо явно вказано вимогу до векторізації.

**Рішення:**
```
"CRITICAL: Use vectorized operations only. NO Python loops over:
- batch_size
- N (number of nodes)
- sequence_length

Use einsum, bmm, gather, or other tensor operations instead."
```

### 6.3. Помилка 3: Ігнорування обмежень пам'яті

**Симптом:** LLM створює проміжні тензори великого розміру.

**Причина:** Не вказано обмеження пам'яті.

**Рішення:**
```
"Memory constraints:
- Maximum intermediate tensor size: [batch_size, N, N, c]
- FORBIDDEN: tensors of size [batch_size, N, N, N, c] or larger
- Use in-place operations where possible (if gradients allow)
- Consider gradient checkpointing for large models"
```

### 6.4. Помилка 4: Неправильна обробка масок

**Симптом:** Маски застосовуються неправильно, що призводить до NaN у softmax.

**Причина:** Не вказано точну семантику маски.

**Рішення:**
```
"Mask specification:
- mask[i, j] = True means position (i, j) is ALLOWED
- mask[i, j] = False means position (i, j) is FORBIDDEN (masked)
- Before softmax: set masked logits to -inf
- Use torch.masked_fill(~mask, float('-inf')) for clarity"
```

---

## 7. Engineering Challenge: AI-Resistant Assessment

### 7.1. Задача: Написати промпт для складного тензорного операції

**Контекст:**
Потрібно реалізувати **Geodesic Attention** (адаптація Invariant Point Attention для 2D простору) з модуля 06.

**Математична формалізація:**
$$\text{GeodesicAttention}(\mathbf{q}, \mathbf{k}, \mathbf{v}, \mathbf{T}) = \text{Softmax}\left(\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}} + \text{GeodesicBias}(\mathbf{T})\right) \mathbf{v}$$

Де:
- $\mathbf{q}, \mathbf{k}, \mathbf{v} \in \mathbb{R}^{N \times d}$ — query, key, value
- $\mathbf{T}_i = (\mathbf{R}_i, \mathbf{t}_i)$ — frame для міста $i$ (обертання + позиція)
- $\text{GeodesicBias}(\mathbf{T}_i, \mathbf{T}_j) = f(d_{ij}, \theta_{ij})$ — bias на основі відстані та кута

**Деталі:**
- $d_{ij} = \|\mathbf{t}_i - \mathbf{t}_j\|_2$ — відстань між позиціями
- $\theta_{ij} = \text{angle}(\mathbf{R}_i, \mathbf{R}_j)$ — кут між орієнтаціями
- $f(d, \theta) = \text{MLP}([d, \cos(\theta), \sin(\theta)])$ — learnable функція

**Вимоги:**
- Підтримка batch dimension
- Memory-efficient (no O(N^3) tensors)
- Differentiable
- Type hints + docstring з LaTeX

**Ваше завдання:**

1. **Напишіть повний промпт** для генерації цієї функції
2. **Обґрунтуйте структуру:**
   - Чому саме такий порядок компонентів?
   - Які деталі критичні, а які можна опустити?
   - Як перевірити, що LLM зрозумів правильно?

3. **Оцініть якість результату:**
   - Як перевірити математичну коректність?
   - Як перевірити memory efficiency?
   - Які тести написати для валідації?

**Критерії оцінки:**
- **Недостатньо:** "Implement geodesic attention" (немає деталей)
- **Добре:** Промпт з математикою та tensor specs
- **Відмінно:** Структурований промпт з обґрунтуванням, план тестування, аналіз потенційних помилок

### 7.2. Референсне рішення (для викладача)

**Повний промпт:**

```
You are an expert PyTorch developer specializing in geometric deep learning 
and invariant attention mechanisms. You implement mathematically precise, 
memory-efficient tensor operations.

Implement Geodesic Attention (2D adaptation of Invariant Point Attention) 
for TSP/VRP routing:

Mathematical definition:
GeodesicAttention(q, k, v, T) = Softmax(QK^T / √d_k + GeodesicBias(T)) V

Where:
- q, k, v ∈ ℝ^(N×d): query, key, value embeddings
- T_i = (R_i, t_i): frame for city i
  - R_i ∈ SO(2): 2D rotation matrix [2, 2]
  - t_i ∈ ℝ^2: position [2]
- GeodesicBias(T_i, T_j) = MLP([d_ij, cos(θ_ij), sin(θ_ij)])
  - d_ij = ||t_i - t_j||_2: Euclidean distance
  - θ_ij = angle(R_i, R_j): angle between orientations

Tensor specifications:
Inputs:
- q: [batch_size, N, d] - query embeddings
- k: [batch_size, N, d] - key embeddings  
- v: [batch_size, N, d] - value embeddings
- frames: dict with keys:
  - 'rotation': [batch_size, N, 2, 2] - rotation matrices
  - 'translation': [batch_size, N, 2] - positions

Output:
- attn_output: [batch_size, N, d] - attention output

Memory constraints:
- NO intermediate tensors larger than [batch_size, N, N, d]
- Compute bias matrix [batch_size, N, N] efficiently
- Use einsum for attention computation

Code requirements:
- Type hints for all inputs/outputs
- Docstring with LaTeX math notation
- Inline comments explaining geometric operations
- Support both training and inference modes
```

**Обґрунтування структури:**

1. **Role:** Встановлює експертність та домен
2. **Mathematics:** Повна формалізація з усіма деталями
3. **Tensor Specs:** Точні розмірності, включаючи структуру frames
4. **Memory:** Явні обмеження
5. **Code Style:** Вимоги до документації

**Очікуваний результат:**

```python
def geodesic_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    frames: dict,
    d_k: int = None
) -> torch.Tensor:
    """
    Geodesic Attention for 2D routing problems.
    
    Computes attention with geometric bias based on frames:
    Attention = Softmax(QK^T / √d_k + GeodesicBias(T)) V
    
    Where GeodesicBias depends on distance and angle between frames.
    
    Args:
        q: Query embeddings [batch_size, N, d]
        k: Key embeddings [batch_size, N, d]
        v: Value embeddings [batch_size, N, d]
        frames: Dict with 'rotation' [batch_size, N, 2, 2] and 
                'translation' [batch_size, N, 2]
        d_k: Dimension of key (default: d)
    
    Returns:
        Attention output [batch_size, N, d]
    """
    batch_size, N, d = q.shape
    if d_k is None:
        d_k = d
    
    R = frames['rotation']  # [batch_size, N, 2, 2]
    t = frames['translation']  # [batch_size, N, 2]
    
    # Compute distances: d_ij = ||t_i - t_j||_2
    # t_i: [batch_size, N, 2], t_j: [batch_size, N, 2]
    # Expand for pairwise: [batch_size, N, 1, 2] - [batch_size, 1, N, 2]
    t_i = t.unsqueeze(2)  # [batch_size, N, 1, 2]
    t_j = t.unsqueeze(1)  # [batch_size, 1, N, 2]
    t_diff = t_i - t_j  # [batch_size, N, N, 2]
    distances = torch.norm(t_diff, dim=-1)  # [batch_size, N, N]
    
    # Compute angles: θ_ij = angle(R_i, R_j)
    # For 2D rotations: R = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    # Angle from rotation matrix: θ = atan2(R[1,0], R[0,0])
    angles_i = torch.atan2(R[:, :, 1, 0], R[:, :, 0, 0])  # [batch_size, N]
    angles_j = torch.atan2(R[:, :, 1, 0], R[:, :, 0, 0])  # [batch_size, N]
    
    # Pairwise angle differences
    angles_i = angles_i.unsqueeze(2)  # [batch_size, N, 1]
    angles_j = angles_j.unsqueeze(1)  # [batch_size, 1, N]
    angle_diffs = angles_i - angles_j  # [batch_size, N, N]
    
    # Geodesic bias: MLP([d_ij, cos(θ_ij), sin(θ_ij)])
    bias_features = torch.stack([
        distances,
        torch.cos(angle_diffs),
        torch.sin(angle_diffs)
    ], dim=-1)  # [batch_size, N, N, 3]
    
    # Learnable MLP for bias (simplified: linear layer)
    # In practice, this would be a proper MLP
    bias = self.bias_mlp(bias_features).squeeze(-1)  # [batch_size, N, N]
    
    # Attention scores: QK^T / √d_k
    scores = torch.einsum('bnd,bmd->bnm', q, k) / math.sqrt(d_k)
    
    # Add geometric bias
    scores = scores + bias
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)  # [batch_size, N, N]
    
    # Apply to values
    attn_output = torch.einsum('bnm,bmd->bnd', attn_weights, v)
    
    return attn_output
```

**План тестування:**

1. **Математична коректність:**
   - Перевірити, що bias інваріантний до глобального обертання
   - Перевірити, що attention weights сумуються до 1

2. **Memory efficiency:**
   - Перевірити, що немає тензорів розміру [batch_size, N, N, N, ...]
   - Профілювати пам'ять на великих батчах

3. **Градієнти:**
   - Перевірити, що всі операції диференційовані
   - Перевірити backward pass на реальних даних

---

## 8. Джерела та Література

### 8.1. Промпт-інжиніринг та LLM
* **Книга:** *J. White, et al. "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT".* [arXiv:2302.11382](https://arxiv.org/abs/2302.11382) — Каталог патернів промптів.
* **Стаття:** *Reynolds, L., & McDonell, K. (2021). "Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm".* [CHI 2021](https://arxiv.org/abs/2102.07350) — Методологія промпт-програмування.
* **Ресурс:** [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — Офіційний гайд від OpenAI.

### 8.2. Tensor Operations та PyTorch
* **Документація:** [PyTorch Einsum Documentation](https://pytorch.org/docs/stable/generated/torch.einsum.html) — Офіційна документація einsum.
* **Стаття:** *Harris, C. R., et al. (2020). "Array programming with NumPy".* [Nature 2020](https://www.nature.com/articles/s41586-020-2649-2) — Фундаментальні концепції array programming.
* **Ресурс:** [PyTorch Best Practices](https://pytorch.org/docs/stable/notes/best_practices.html) — Рекомендації з оптимізації.

### 8.3. JAX та Functional Programming
* **Документація:** [JAX Documentation](https://jax.readthedocs.io/) — Офіційна документація JAX.
* **Стаття:** *Bradbury, J., et al. (2018). "JAX: Composable transformations of Python+NumPy programs".* [GitHub](https://github.com/google/jax) — Вступ до JAX.
* **Ресурс:** [JAX Tutorials](https://jax.readthedocs.io/en/latest/tutorials/) — Офіційні туторіали.

---

**Наступний крок:** Створення синтетичного бенчмарку ([09_synthetic_benchmark.md](./09_synthetic_benchmark.md)).

