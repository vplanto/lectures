---
title: "13 Evacuation Workshop"
type: lecture
module: Семінар
prerequisites: module 12
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# 13_evacuation_workshop.md: Практикум: Евакуація зі стадіону — A* vs GDL

**Курс:** Геометричний Deep Learning в Логістиці
**Модуль:** Практикум
**Рівень:** Intermediate / Advanced
**Тривалість:** 2-3 години

---

## 1. Постановка задачі: Евакуація зі стадіону

### 1.1. Сценарій

**Задача:** Евакуація 10,000 пішоходів зі стадіону через 4 виходи.

**Геометрія стадіону:**
- **Форма:** Овальна арена з трибунами навколо
- **Розміри:** 200m × 150m (овал)
- **Входи/виходи:** 4 виходи розташовані на північ, південь, схід, захід
- **Пішоходи:** 10,000 пішоходів розподілені по трибунах

**Візуалізація:**

```
        Північ (вихід 1)
            ↑
            │
    Захід  │  Схід
 (вихід 4) │  (вихід 2)
            │
            ↓
        Південь (вихід 3)

    ┌─────────────────┐
    │                 │
    │   Трибуни       │
    │   (10,000       │
    │    пішоходів)   │
    │                 │
    └─────────────────┘
```

**Мета:** Мінімізувати час евакуації та уникнути заторів.

### 1.2. Математична формалізація

**Вхідні дані:**
- Координати пішоходів: $\mathbf{P} = \{(x_i, y_i)\}_{i=1}^{10,000}$
- Координати виходів: $\mathbf{E} = \{(x_j^{exit}, y_j^{exit})\}_{j=1}^{4}$
- Пропускна здатність виходів: $C_j = 50$ пішоходів/хвилину для кожного виходу

**Вихідні дані:**
- Маршрути для кожного пішохода: $\pi_i = (p_i, e_{target}, \text{path})$
- Час евакуації: $T_{total}$

**Обмеження:**
- Кожен пішохід має досягти одного з виходів
- Пропускна здатність виходів обмежена
- Уникнути зіткнень між пішоходами

---

## 2. Рішення через A* (Класичний підхід)

### 2.1. Алгоритм A*

**Ідея:** Для кожного пішохода знайти найкоротший шлях до найближчого виходу.

**Алгоритм:**

```python
def astar_evacuation(pedestrians, exits):
    """
    A* для евакуації: кожен пішохід йде до найближчого виходу.
    """
    routes = {}
    
    for ped_id, (x, y) in pedestrians.items():
        # Знаходимо найближчий вихід
        nearest_exit = min(exits, 
                          key=lambda e: distance((x, y), e))
        
        # A* пошук найкоротшого шляху
        path = astar_search((x, y), nearest_exit, obstacles=[])
        routes[ped_id] = {
            'exit': nearest_exit,
            'path': path,
            'distance': path_length(path)
        }
    
    return routes
```

**Евристика:** Відстань по прямій до виходу: $h(n) = \|(x_n, y_n) - (x_{exit}, y_{exit})\|$

**Функція вартості:** $f(n) = g(n) + h(n)$, де $g(n)$ — відстань від початку до $n$.

### 2.2. Результат A*

**Що відбувається:**

1. Кожен пішохід обчислює найкоротший шлях до найближчого виходу
2. Всі пішоходи з північної частини йдуть до північного виходу
3. Всі пішоходи з південної частини йдуть до південного виходу
4. Аналогічно для східного та західного виходів

**Проблема:**

**Затори біля виходів!**

- Північний вихід: 3,000 пішоходів намагаються пройти через пропускну здатність 50/хв
- Час очікування: $3,000 / 50 = 60$ хвилин
- Аналогічно для інших виходів

**Візуалізація:**

```
        Північ (вихід 1)
            ↑
            │ ════════════
            │ ════════════  ← Затор! 3,000 пішоходів
            │ ════════════
            │
    Захід  │  Схід
 (вихід 4) │  (вихід 2)
            │
            ↓ ════════════
            │ ════════════  ← Затор! 2,500 пішоходів
            │ ════════════
        Південь (вихід 3)
```

**Час евакуації:**

$$T_{A*} = \max_j \frac{N_j}{C_j}$$

Де $N_j$ — кількість пішоходів, які обирають вихід $j$.

**Для нашого прикладу:**
- $N_1 = 3,000$ (північний вихід)
- $N_2 = 2,500$ (східний вихід)
- $N_3 = 2,500$ (південний вихід)
- $N_4 = 2,000$ (західний вихід)

$$T_{A*} = \max\left(\frac{3,000}{50}, \frac{2,500}{50}, \frac{2,500}{50}, \frac{2,000}{50}\right) = 60 \text{ хвилин}$$

### 2.3. Обмеження A*

**Проблеми:**

1. **Локальна оптимізація:** Кожен пішохід оптимізує тільки свій шлях, не враховуючи інших
2. **Відсутність координації:** Немає механізму для розподілу навантаження
3. **Затори:** Всі пішоходи з однієї зони йдуть до одного виходу
4. **Висока "енергія" біля виходів:** Концентрація пішоходів створює "тиск"

**Математично:**

A* мінімізує для кожного пішохода $i$:

$$\min_{\pi_i} d(\pi_i)$$

Де $d(\pi_i)$ — відстань маршруту пішохода $i$.

**Але не мінімізує:**

$$\min_{\{\pi_i\}} \max_j \frac{|\{i : \pi_i \text{ веде до виходу } j\}|}{C_j}$$

Тобто, A* не враховує **глобальну енергію системи**.

---

## 3. Рішення через GDL (Агентне моделювання)

### 3.1. Агентна інтерпретація

**Ключова ідея:** Кожен пішохід — це **автономний агент** з власною "сенсорною системою" (Geodesic Attention), яка дозволяє йому "бачити" інших пішоходів та виходи.

**Аналогія з AlphaFold 2:**

- **Атоми (білок)** → **Пішоходи (стадіон)**
- **Енергія взаємодії** → **"Тиск" (енергія) біля виходів**
- **Згортання білка** → **Евакуація пішоходів**
- **IPA (Invariant Point Attention)** → **Geodesic Attention**

### 3.2. Математична модель

**Стан агента (пішохода $i$):**

$$\mathbf{T}_i = (x_i, y_i, \theta_i, e_i)$$

Де:
- $(x_i, y_i)$ — позиція
- $\theta_i$ — напрямок руху
- $e_i$ — "енергія" (час очікування, відстань до виходу)

**Енергія системи:**

$$E(\{\mathbf{T}_i\}) = \sum_{i} E_{distance}(\mathbf{T}_i) + \sum_{j} E_{pressure}(\text{exit}_j) + \sum_{i,j} E_{collision}(\mathbf{T}_i, \mathbf{T}_j)$$

Де:
- $E_{distance}$ — енергія відстані (мінімізуємо довжину шляху)
- $E_{pressure}$ — енергія "тиску" біля виходів (висока, коли багато пішоходів)
- $E_{collision}$ — енергія зіткнень (штрафує за близькість між пішоходами)

**Енергія "тиску" біля виходу:**

$$E_{pressure}(\text{exit}_j) = \lambda_{pressure} \cdot \left(\frac{N_j}{C_j}\right)^2$$

Де:
- $N_j$ — кількість пішоходів біля виходу $j$
- $C_j$ — пропускна здатність виходу $j$
- $\lambda_{pressure}$ — вага штрафу за "тиск"

**Інтерпретація:**

- Якщо $N_j \ll C_j$: $E_{pressure} \approx 0$ (немає тиску)
- Якщо $N_j \approx C_j$: $E_{pressure}$ середня (нормальне навантаження)
- Якщо $N_j \gg C_j$: $E_{pressure}$ висока (великий тиск, затор)

### 3.3. Geodesic Attention для координації

**Механізм:**

Кожен пішохід $i$ обчислює attention ваги до виходів:

$$\alpha_{ij} = \text{Softmax}\left(\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + \text{GeodesicBias}(\mathbf{T}_i, \mathbf{E}_j) + \beta \cdot E_{pressure}(\text{exit}_j)\right)$$

Де:
- $\mathbf{q}_i$ — query пішохода $i$ ("Куди мені йти?")
- $\mathbf{k}_j$ — key виходу $j$ (інформація про вихід)
- $\text{GeodesicBias}$ — геометричний bias (відстань, кут)
- $E_{pressure}(\text{exit}_j)$ — енергія "тиску" біля виходу $j$

**Вплив енергії "тиску":**

- **Високий тиск** → високий $E_{pressure}$ → низький $\alpha_{ij}$ → пішохід уникає цього виходу
- **Низький тиск** → низький $E_{pressure}$ → високий $\alpha_{ij}$ → пішохід обирає цей вихід

**Результат:**

Пішоходи **автоматично розподіляються** між виходами, уникаючи зон високого тиску!

### 3.4. Ітеративна оптимізація (Recycling)

**Time-step 0 ($t=0$):**

Початкові стани пішоходів:
- $\mathbf{T}_i^{(0)} = (x_i, y_i, \theta_i^{(0)}, e_i^{(0)})$

**Time-step 1 ($t=1$):**

Обчислюємо attention ваги та оновлюємо стани:

$$\mathbf{T}_i^{(1)} = \mathbf{T}_i^{(0)} + \Delta t \cdot \sum_{j} \alpha_{ij}^{(0)} \mathbf{v}_j^{(0)}$$

Де $\mathbf{v}_j^{(0)}$ — напрямок до виходу $j$ з урахуванням енергії тиску.

**Time-step 2 ($t=2$):**

Аналогічно, використовуючи оновлені стани:

$$\mathbf{T}_i^{(2)} = \mathbf{T}_i^{(1)} + \Delta t \cdot \sum_{j} \alpha_{ij}^{(1)} \mathbf{v}_j^{(1)}$$

**Ітерація:**

Після $T$ time-steps, система досягає **рівноваги**:
- Пішоходи розподілені рівномірно між виходами
- Енергія "тиску" мінімізована
- Час евакуації оптимізований

### 3.5. Результат GDL

**Що відбувається:**

1. **Початковий стан:** Пішоходи розкидані по трибунах
2. **Обчислення енергії:** Система обчислює енергію "тиску" біля кожного виходу
3. **Geodesic Attention:** Кожен пішохід "бачить" тиск біля виходів через attention ваги
4. **Автоматичний розподіл:** Пішоходи автоматично розподіляються, уникаючи зон високого тиску
5. **Рівномірний потік:** Кожен вихід обслуговує приблизно $10,000 / 4 = 2,500$ пішоходів

**Візуалізація:**

```
        Північ (вихід 1)
            ↑
            │ ────────
            │ ────────  ← Рівномірний потік
            │ ────────
            │
    Захід  │  Схід
 (вихід 4) │  (вихід 2)
            │ ────────
            │ ────────  ← Рівномірний потік
            │ ────────
            │
        Південь (вихід 3)
```

**Час евакуації:**

$$T_{GDL} = \frac{N_{total}}{4 \cdot C} = \frac{10,000}{4 \times 50} = 50 \text{ хвилин}$$

**Покращення:** $60 - 50 = 10$ хвилин (16.7% швидше)

---

## 4. Детальне порівняння

> **📚 Порівняння ефективності:** Порівняння ефективності A* та GDL базується на результатах, отриманих у модулі порівняння рішень [10_solution_comparison.md](./10_solution_comparison.md).

### 4.1. Таблиця порівняння

| Аспект | A* (Класичний) | GDL (Агентне моделювання) |
|--------|----------------|---------------------------|
| **Підхід** | Локальна оптимізація для кожного пішохода | Глобальна оптимізація системи |
| **Координація** | ❌ Відсутня | ✅ Через Geodesic Attention |
| **Врахування тиску** | ❌ Ні | ✅ Так (через $E_{pressure}$) |
| **Розподіл навантаження** | ❌ Неравномірний | ✅ Рівномірний |
| **Час евакуації** | 60 хвилин | 50 хвилин |
| **Затори** | ✅ Виникають | ❌ Уникаються |
| **Масштабованість** | $O(N \cdot \log N)$ | $O(N^2)$ (але паралелізується) |
| **Реальний час** | ⚠️ Може бути повільним | ✅ Швидко на GPU |

### 4.2. Візуалізація енергетичного ландшафту

**A* (Локальна оптимізація):**

```
Енергія E(x)
    ↑
    |     ╱╲        ╱╲
    |    ╱  ╲      ╱  ╲
    |   ╱    ╲    ╱    ╲
    |  ╱      ╲  ╱      ╲
    | ╱        ╲╱        ╲
    |╱          ╲          ╲
    └──────────────────────────→ Позиція x
         Високі піки (затори)
```

**GDL (Глобальна оптимізація):**

```
Енергія E(x)
    ↑
    |     ╱╲
    |    ╱  ╲
    |   ╱    ╲
    |  ╱      ╲
    | ╱        ╲
    |╱          ╲
    └──────────────────────────→ Позиція x
         Плавний спуск (рівномірний потік)
```

### 4.3. Аналіз "тиску" біля виходів

**A* (Час $t=0$):**

```
Вихід 1 (Північ):  N₁ = 3,000  →  E_pressure = λ · (3000/50)² = 3600λ
Вихід 2 (Схід):    N₂ = 2,500  →  E_pressure = λ · (2500/50)² = 2500λ
Вихід 3 (Південь): N₃ = 2,500  →  E_pressure = λ · (2500/50)² = 2500λ
Вихід 4 (Захід):   N₄ = 2,000  →  E_pressure = λ · (2000/50)² = 1600λ
```

**Максимальний тиск:** $E_{max} = 3600\lambda$ (північний вихід)

**GDL (Час $t=T$, після оптимізації):**

```
Вихід 1 (Північ):  N₁ = 2,500  →  E_pressure = λ · (2500/50)² = 2500λ
Вихід 2 (Схід):    N₂ = 2,500  →  E_pressure = λ · (2500/50)² = 2500λ
Вихід 3 (Південь): N₃ = 2,500  →  E_pressure = λ · (2500/50)² = 2500λ
Вихід 4 (Захід):   N₄ = 2,500  →  E_pressure = λ · (2500/50)² = 2500λ
```

**Максимальний тиск:** $E_{max} = 2500\lambda$ (рівномірний)

**Зменшення тиску:** $(3600 - 2500) / 3600 = 30.6\%$

---

## 5. Практична реалізація

### 5.1. Реалізація A*

```python
import heapq
from math import sqrt

def astar_search(start, goal, obstacles):
    """
    A* пошук найкоротшого шляху.
    """
    def heuristic(pos):
        return sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
    
    def distance(pos1, pos2):
        return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # Відновлюємо шлях
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Перевіряємо сусідів
        for neighbor in get_neighbors(current, obstacles):
            tentative_g = g_score[current] + distance(current, neighbor)
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # Шлях не знайдено

def astar_evacuation(pedestrians, exits):
    """
    A* для евакуації: кожен пішохід йде до найближчого виходу.
    """
    routes = {}
    
    for ped_id, pos in pedestrians.items():
        # Знаходимо найближчий вихід
        nearest_exit = min(exits, 
                          key=lambda e: distance(pos, e))
        
        # A* пошук найкоротшого шляху
        path = astar_search(pos, nearest_exit, obstacles=[])
        routes[ped_id] = {
            'exit': nearest_exit,
            'path': path,
            'distance': sum(distance(path[i], path[i+1]) 
                          for i in range(len(path)-1))
        }
    
    return routes
```

### 5.2. Реалізація GDL

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class EvacuationGDL(nn.Module):
    """
    GDL модель для евакуації зі стадіону.
    """
    def __init__(self, num_exits=4, d_model=128):
        super().__init__()
        self.num_exits = num_exits
        self.d_model = d_model
        
        # Embeddings для пішоходів та виходів
        self.pedestrian_embedding = nn.Linear(4, d_model)  # (x, y, θ, e)
        self.exit_embedding = nn.Linear(2, d_model)  # (x, y)
        
        # Geodesic Attention
        self.attention = GeodesicAttention(d_model)
        
        # MLP для обчислення напрямку руху
        self.direction_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)  # (Δx, Δy)
        )
    
    def forward(self, pedestrian_states, exit_positions, delta_t=0.1):
        """
        Args:
            pedestrian_states: [N, 4] - (x, y, θ, e) для кожного пішохода
            exit_positions: [M, 2] - (x, y) для кожного виходу
            delta_t: крок симуляції
        
        Returns:
            new_states: [N, 4] - нові стани пішоходів
            attention_weights: [N, M] - attention ваги між пішоходами та виходами
        """
        N = pedestrian_states.shape[0]
        M = exit_positions.shape[0]
        
        # Обчислюємо embeddings
        ped_emb = self.pedestrian_embedding(pedestrian_states)  # [N, d_model]
        exit_emb = self.exit_embedding(exit_positions)  # [M, d_model]
        
        # Обчислюємо енергію "тиску" біля виходів
        pressure_energies = self.compute_pressure_energy(
            pedestrian_states, exit_positions
        )  # [M]
        
        # Geodesic Attention з урахуванням тиску
        attention_weights = self.attention(
            ped_emb, exit_emb, 
            pedestrian_states[:, :2], exit_positions,
            pressure_energies
        )  # [N, M]
        
        # Обчислюємо напрямок руху для кожного пішохода
        directions = self.direction_mlp(ped_emb)  # [N, 2]
        
        # Оновлюємо стани
        new_positions = pedestrian_states[:, :2] + delta_t * directions
        new_states = torch.cat([
            new_positions,
            pedestrian_states[:, 2:4]  # Зберігаємо θ та e
        ], dim=1)
        
        return new_states, attention_weights
    
    def compute_pressure_energy(self, pedestrian_states, exit_positions):
        """
        Обчислює енергію "тиску" біля кожного виходу.
        """
        N = pedestrian_states.shape[0]
        M = exit_positions.shape[0]
        
        # Відстані від пішоходів до виходів
        ped_pos = pedestrian_states[:, :2].unsqueeze(1)  # [N, 1, 2]
        exit_pos = exit_positions.unsqueeze(0)  # [1, M, 2]
        distances = torch.norm(ped_pos - exit_pos, dim=2)  # [N, M]
        
        # Кількість пішоходів біля кожного виходу (в радіусі 10m)
        radius = 10.0
        near_exit = (distances < radius).float()  # [N, M]
        N_j = near_exit.sum(dim=0)  # [M]
        
        # Пропускна здатність (50 пішоходів/хвилину)
        C = 50.0
        
        # Енергія тиску
        pressure_energies = (N_j / C) ** 2  # [M]
        
        return pressure_energies


class GeodesicAttention(nn.Module):
    """
    Geodesic Attention з урахуванням енергії тиску.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        
        # MLP для GeodesicBias
        self.geodesic_bias_mlp = nn.Sequential(
            nn.Linear(3, 32),  # (distance, cos(θ), sin(θ))
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, ped_emb, exit_emb, ped_pos, exit_pos, pressure_energies):
        """
        Args:
            ped_emb: [N, d_model] - embeddings пішоходів
            exit_emb: [M, d_model] - embeddings виходів
            ped_pos: [N, 2] - позиції пішоходів
            exit_pos: [M, 2] - позиції виходів
            pressure_energies: [M] - енергія тиску біля виходів
        
        Returns:
            attention_weights: [N, M]
        """
        N, M = ped_emb.shape[0], exit_emb.shape[0]
        
        # Query та Key
        Q = self.query_proj(ped_emb)  # [N, d_model]
        K = self.key_proj(exit_emb)  # [M, d_model]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(0, 1)) / sqrt(self.d_model)  # [N, M]
        
        # Geodesic Bias
        ped_pos_expanded = ped_pos.unsqueeze(1)  # [N, 1, 2]
        exit_pos_expanded = exit_pos.unsqueeze(0)  # [1, M, 2]
        diff = ped_pos_expanded - exit_pos_expanded  # [N, M, 2]
        
        distances = torch.norm(diff, dim=2)  # [N, M]
        angles = torch.atan2(diff[:, :, 1], diff[:, :, 0])  # [N, M]
        
        geodesic_features = torch.stack([
            distances,
            torch.cos(angles),
            torch.sin(angles)
        ], dim=2)  # [N, M, 3]
        
        geodesic_bias = self.geodesic_bias_mlp(geodesic_features).squeeze(2)  # [N, M]
        
        # Pressure bias (високий тиск → низький attention)
        pressure_bias = -pressure_energies.unsqueeze(0)  # [1, M]
        
        # Загальний bias
        total_bias = geodesic_bias + pressure_bias  # [N, M]
        
        # Softmax
        attention_weights = F.softmax(scores + total_bias, dim=1)  # [N, M]
        
        return attention_weights


def gdl_evacuation(pedestrians, exits, num_steps=100, delta_t=0.1):
    """
    GDL для евакуації: агентне моделювання з урахуванням тиску.
    """
    model = EvacuationGDL(num_exits=len(exits))
    
    # Конвертуємо в тензори
    ped_states = torch.tensor([
        [x, y, 0.0, 0.0]  # (x, y, θ, e)
        for (x, y) in pedestrians.values()
    ])
    exit_pos = torch.tensor(exits)
    
    history = []
    
    for step in range(num_steps):
        # Forward pass
        new_states, attention_weights = model(ped_states, exit_pos, delta_t)
        
        # Зберігаємо історію
        history.append({
            'states': new_states.clone(),
            'attention': attention_weights.clone()
        })
        
        # Оновлюємо стани
        ped_states = new_states
    
    return history
```

### 5.3. Порівняння результатів

```python
# Генерація тестових даних
import numpy as np

# Стадіон: овал 200m × 150m
def generate_stadium_pedestrians(n=10000):
    """Генерує позиції пішоходів на трибунах."""
    pedestrians = {}
    for i in range(n):
        # Випадкові координати в овалі
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(50, 100)  # Радіус від центру
        x = 100 + r * np.cos(angle)
        y = 75 + r * np.sin(angle)
        pedestrians[i] = (x, y)
    return pedestrians

# Виходи
exits = [
    (100, 0),    # Південь
    (100, 150),  # Північ
    (0, 75),     # Захід
    (200, 75)    # Схід
]

# Генерація даних
pedestrians = generate_stadium_pedestrians(10000)

# A* рішення
routes_astar = astar_evacuation(pedestrians, exits)
time_astar = compute_evacuation_time(routes_astar, exits)

# GDL рішення
history_gdl = gdl_evacuation(pedestrians, exits, num_steps=100)
time_gdl = compute_evacuation_time_from_history(history_gdl, exits)

# Порівняння
print(f"A* час евакуації: {time_astar:.1f} хвилин")
print(f"GDL час евакуації: {time_gdl:.1f} хвилин")
print(f"Покращення: {((time_astar - time_gdl) / time_astar * 100):.1f}%")
```

---

## 6. Візуалізація та аналіз

### 6.1. Візуалізація потоків

**A* (Локальна оптимізація):**

```
        Північ
            ↑
            │ ████████████
            │ ████████████  ← Концентрація
            │ ████████████
            │
    Захід  │  Схід
            │ ████████████
            │ ████████████  ← Концентрація
            │ ████████████
            │
        Південь
```

**GDL (Глобальна оптимізація):**

```
        Північ
            ↑
            │ ░░░░░░░░░░░░
            │ ░░░░░░░░░░░░  ← Рівномірний потік
            │ ░░░░░░░░░░░░
            │
    Захід  │  Схід
            │ ░░░░░░░░░░░░
            │ ░░░░░░░░░░░░  ← Рівномірний потік
            │ ░░░░░░░░░░░░
            │
        Південь
```

### 6.2. Графік енергії "тиску" в часі

**A*:**

```
E_pressure
    ↑
3600│ ████████████████████████████████████
    │
2500│ ████████████████████████████████████
    │
1600│ ████████████████████████████████████
    │
    └────────────────────────────────────→ Час
    0                                   60 хв
    (Постійний високий тиск)
```

**GDL:**

```
E_pressure
    ↑
3600│ ████
    │    ╲
2500│     ╲═══════════════════════════════
    │      ╲
1600│       ╲
    │
    └────────────────────────────────────→ Час
    0                                   50 хв
    (Тиск зменшується до рівномірного)
```

### 6.3. Аналіз розподілу пішоходів

**A* (Час $t=0$):**

```
Вихід 1: ████████████████████████████████████ 3000 пішоходів (30%)
Вихід 2: ████████████████████████████████     2500 пішоходів (25%)
Вихід 3: ████████████████████████████████     2500 пішоходів (25%)
Вихід 4: ████████████████████████████         2000 пішоходів (20%)
```

**GDL (Час $t=T$):**

```
Вихід 1: ████████████████████████████████████ 2500 пішоходів (25%)
Вихід 2: ████████████████████████████████████ 2500 пішоходів (25%)
Вихід 3: ████████████████████████████████████ 2500 пішоходів (25%)
Вихід 4: ████████████████████████████████████ 2500 пішоходів (25%)
```

**Результат:** Рівномірний розподіл! ✅

---

## 7. Висновки та уроки

### 7.1. Ключові висновки

**A* (Класичний підхід):**
- ✅ **Простота:** Легко реалізувати та зрозуміти
- ✅ **Швидкість:** Швидкий для окремих агентів
- ❌ **Локальна оптимізація:** Не враховує глобальну енергію системи
- ❌ **Затори:** Створює затори біля виходів
- ❌ **Неефективність:** Час евакуації не оптимальний

**GDL (Агентне моделювання):**
- ✅ **Глобальна оптимізація:** Враховує енергію всієї системи
- ✅ **Координація:** Автоматична координація через Attention
- ✅ **Рівномірний розподіл:** Уникає заторів
- ✅ **Ефективність:** Оптимальний час евакуації
- ⚠️ **Складність:** Складніша реалізація
- ⚠️ **Обчислювальна складність:** Потребує GPU для великих систем

### 7.2. Аналогія з AlphaFold 2

**Згортання білка:**
- Атоми "відчувають" енергетичний ландшафт
- Автоматично організуються в стабільну структуру
- Уникають зон високої енергії (steric clashes)

**Евакуація пішоходів:**
- Пішоходи "відчувають" енергію "тиску" через Geodesic Attention
- Автоматично розподіляються рівномірно
- Уникають зон високого тиску (заторів)

**Математична еквівалентність:**

Обидва процеси мінімізують енергію системи:
- **Білок:** $E(\mathbf{R}) = E_{bond} + E_{angle} + E_{steric}$
- **Евакуація:** $E(\{\mathbf{T}_i\}) = E_{distance} + E_{pressure} + E_{collision}$

### 7.3. Практичні рекомендації

**Коли використовувати A*:**
- Мала кількість агентів (< 100)
- Прості сценарії без заторів
- Швидке прототипування
- Обмежені обчислювальні ресурси

**Коли використовувати GDL:**
- Велика кількість агентів (> 1,000)
- Складні сценарії з потенційними заторами
- Потрібна глобальна оптимізація
- Доступні GPU ресурси

**Гібридний підхід:**
- Використовувати A* для початкового планування
- Застосовувати GDL для уточнення та координації
- Комбінувати переваги обох підходів

---

## 8. Додаткові вправи

### 8.1. Вправа 1: Модифікація пропускної здатності

**Задача:** Що станеться, якщо один вихід має пропускну здатність 100 пішоходів/хвилину, а інші — 50?

**Питання:**
1. Як A* розподілить пішоходів?
2. Як GDL розподілить пішоходів?
3. Яка різниця в часі евакуації?

### 8.2. Вправа 2: Динамічні перешкоди

**Задача:** Додайте перешкоди (наприклад, закриті секції трибун).

**Питання:**
1. Як A* обійде перешкоди?
2. Як GDL врахує перешкоди через енергію обмежень?
3. Який підхід краще для динамічних перешкод?

### 8.3. Вправа 3: Масштабування

**Задача:** Збільште кількість пішоходів до 100,000.

**Питання:**
1. Чи A* впорається з таким масштабом?
2. Як GDL масштабується?
3. Які оптимізації потрібні для обох підходів?

---

## 9. Додаткові ресурси

### 9.1. Література

- **A* алгоритм:** *Hart, P. E., et al. (1968). "A formal basis for the heuristic determination of minimum cost paths".* — Оригінальна стаття про A*
- **Агентне моделювання:** *Bonabeau, E. (2002). "Agent-based modeling: Methods and techniques for simulating human systems".* — Методи агентного моделювання
- **Евакуація:** *Helbing, D., & Molnár, P. (1995). "Social force model for pedestrian dynamics".* — Фізична модель евакуації

### 9.2. Пов'язані розділи

- [06_tsp_via_af2_geometry.md](./06_tsp_via_af2_geometry.md) — Теорія адаптації AlphaFold 2
- [12_emergent_behavior.md](./12_emergent_behavior.md) — Емерджентна поведінка та енергетичний ландшафт

---

**Кінець практикуму**

