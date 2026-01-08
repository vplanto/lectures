# 09_synthetic_benchmark.md: Створення синтетичного бенчмарку

**Курс:** Геометричний Deep Learning в Логістиці
**Модуль:** 4. Практикум (Workshop)
**Рівень:** Advanced / Expert

---

## 1. Реальний тригер: Чому реальні дані не підходять для навчання

### 1.1. Production Case: Overfitting на реальних даних (2023)

У 2023 році команда намагалася навчити модель VRP на реальних даних доставки з міста. Модель показала 95% точність на тренувальному сеті, але лише 60% на тестовому.

**Технічний розбір:**
- **Тренувальні дані:** 10,000 реальних маршрутів з одного району міста
- **Тестові дані:** 1,000 маршрутів з іншого району
- **Проблема:** Модель запам'ятала географічні особливості району (розташування перехресть, паттерни трафіку), а не навчилася загальним принципам оптимізації

**Корінь проблеми:**
1. **Географічна зміщеність:** Всі тренувальні дані з одного району (подібна структура доріг)
2. **Відсутність варіативності:** Розміри задачі $N$ були в межах 20-30 клієнтів (вузький діапазон)
3. **Неточні ground truth:** Реальні маршрути не завжди оптимальні (залежать від водія, трафіку)

**Висновок:** Реальні дані мають bias та обмежену варіативність. Потрібні синтетичні бенчмарки з контрольованими параметрами та відомими оптимальними рішеннями.

---

## 2. Математична формалізація генерації датасету

### 2.1. Структура екземпляра VRP

**Повний екземпляр VRP визначається як:**
$$\mathcal{I} = (G, \mathbf{X}, \mathbf{D}, \mathbf{C}, \mathbf{Q}, \mathbf{T})$$

Де:
- $G = (V, E)$ — граф міст (вершини та ребра)
- $\mathbf{X} \in \mathbb{R}^{N \times 2}$ — координати міст
- $\mathbf{D} \in \mathbb{R}^{N \times N}$ — матриця відстаней
- $\mathbf{C} \in \mathbb{R}^{N \times N}$ — матриця вартостей
- $\mathbf{Q} \in \mathbb{R}^{N}$ — попит клієнтів
- $\mathbf{T} \in \mathbb{R}^{N \times 2}$ — часові вікна $[e_i, l_i]$

**Мета генерації:** Створити розподіл $p(\mathcal{I})$ екземплярів, який:
1. **Репрезентує реальність:** Географічні розподіли, обмеження
2. **Контрольований:** Параметри можна варіювати систематично
3. **Валідований:** Існують точні оптимальні рішення для малих $N$

### 2.2. Розподіли координат: Від випадкових до структурованих

**Варіант 1: Рівномірний розподіл (Uniform)**
$$x_i, y_i \sim \mathcal{U}(0, 1)$$

**Переваги:**
- Простий для реалізації
- Немає географічного bias

**Недоліки:**
- Не реалістичний (реальні міста мають кластери)
- Легкий для моделі (немає складних паттернів)

**Варіант 2: Кластерний розподіл (Clustered)**
$$c_k \sim \mathcal{U}(0, 1)^2, \quad k = 1, \dots, K$$
$$x_i, y_i \sim \mathcal{N}(c_{k(i)}, \sigma^2 \mathbf{I})$$

Де $k(i)$ — індекс кластера для міста $i$, $\sigma$ — стандартне відхилення.

**Інтерпретація:** Міста згруповані в $K$ районів (кластерів), кожен з центром $c_k$.

**Варіант 3: Міський розподіл (Urban)**
Комбінація кластерів різних розмірів:
- Великі кластери (центр міста): $\sigma_{large} = 0.1$
- Малі кластери (периферія): $\sigma_{small} = 0.05$
- Випадкові точки (окремі клієнти): $p_{random} = 0.2$

**Математично:**
$$x_i, y_i \sim \begin{cases}
\mathcal{N}(c_k, \sigma_{large}^2 \mathbf{I}) & \text{з ймовірністю } p_{center} \\
\mathcal{N}(c_k, \sigma_{small}^2 \mathbf{I}) & \text{з ймовірністю } p_{periphery} \\
\mathcal{U}(0, 1)^2 & \text{з ймовірністю } p_{random}
\end{cases}$$

**Висновок:** Кластерний розподіл краще репрезентує реальність, але потребує більше параметрів.

### 2.3. Метрики відстаней: Евклідова vs Дорожня

**Евклідова відстань:**
$$d_{ij}^{euclidean} = ||\mathbf{x}_i - \mathbf{x}_j||_2 = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

**Переваги:**
- Швидко обчислюється: $O(N^2)$
- Диференційована (для gradient-based оптимізації)
- Симетрична: $d_{ij} = d_{ji}$

**Недоліки:**
- Не враховує реальні дороги (може бути гора, річка)
- Завжди менша за дорожню відстань

**Дорожня відстань (Manhattan для міста):**
$$d_{ij}^{manhattan} = |x_i - x_j| + |y_i - y_j|$$

**Інтерпретація:** Моделює міську сітку доріг (тільки горизонтальні та вертикальні рухи).

**Загальна метрика (Minkowski):**
$$d_{ij}^p = (|x_i - x_j|^p + |y_i - y_j|^p)^{1/p}$$

Де:
- $p=2$: Евклідова
- $p=1$: Manhattan
- $p \to \infty$: Chebyshev ($\max(|x_i - x_j|, |y_i - y_j|)$)

**Вибір для синтетичного бенчмарку:**
Для початку використовуємо **евклідову відстань** (простота), але додаємо **noise** для моделювання неідеальності:
$$d_{ij} = d_{ij}^{euclidean} \cdot (1 + \epsilon_{ij}), \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma_{noise}^2)$$

Де $\sigma_{noise} = 0.05-0.1$ (5-10% варіація).

---

## 3. Генерація координат міст

### 3.1. Базовий генератор (Uniform)

```python
import torch
import numpy as np
from typing import Tuple

def generate_uniform_cities(
    num_cities: int,
    seed: int = None
) -> torch.Tensor:
    """
    Generate cities with uniform random coordinates.
    
    Args:
        num_cities: Number of cities (including depot)
        seed: Random seed for reproducibility
    
    Returns:
        Coordinates [num_cities, 2] in range [0, 1]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Uniform distribution in [0, 1]^2
    coords = torch.rand(num_cities, 2)
    
    # First city is depot (can be fixed at origin)
    coords[0] = torch.tensor([0.0, 0.0])
    
    return coords
```

**Складність:** $O(N)$ — лінійна.

**Пам'ять:** $O(N)$ — один тензор розміру $[N, 2]$.

### 3.2. Кластерний генератор

```python
def generate_clustered_cities(
    num_cities: int,
    num_clusters: int = 3,
    cluster_std: float = 0.1,
    seed: int = None
) -> torch.Tensor:
    """
    Generate cities clustered around centers.
    
    Args:
        num_cities: Total number of cities
        num_clusters: Number of cluster centers
        cluster_std: Standard deviation of cities around cluster center
        seed: Random seed
    
    Returns:
        Coordinates [num_cities, 2]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate cluster centers
    cluster_centers = torch.rand(num_clusters, 2)
    
    # Assign cities to clusters
    cities_per_cluster = num_cities // num_clusters
    remainder = num_cities % num_clusters
    
    coords = []
    
    # Depot at origin
    coords.append(torch.tensor([0.0, 0.0]))
    
    city_idx = 1
    for k in range(num_clusters):
        # Number of cities in this cluster
        n = cities_per_cluster + (1 if k < remainder else 0)
        
        # Generate cities around cluster center
        cluster_coords = torch.randn(n, 2) * cluster_std + cluster_centers[k]
        
        # Clip to [0, 1]
        cluster_coords = torch.clamp(cluster_coords, 0.0, 1.0)
        
        coords.append(cluster_coords)
        city_idx += n
    
    coords = torch.cat(coords, dim=0)[:num_cities]
    
    return coords
```

**Математична формалізація:**
Для кожного кластера $k$:
$$\mathbf{c}_k \sim \mathcal{U}(0, 1)^2$$
$$\mathbf{x}_i \sim \mathcal{N}(\mathbf{c}_k, \sigma^2 \mathbf{I}), \quad \text{для } i \in \text{cluster}_k$$

**Складність:** $O(N)$ — лінійна.

**Пам'ять:** $O(N + K)$ — координати + центри кластерів.

### 3.3. Міський генератор (Urban Distribution)

```python
def generate_urban_cities(
    num_cities: int,
    num_centers: int = 2,
    num_periphery: int = 3,
    center_std: float = 0.15,
    periphery_std: float = 0.05,
    random_ratio: float = 0.2,
    seed: int = None
) -> torch.Tensor:
    """
    Generate cities with urban distribution:
    - Large clusters (city centers)
    - Small clusters (periphery)
    - Random points (scattered customers)
    
    Args:
        num_cities: Total number of cities
        num_centers: Number of city center clusters
        num_periphery: Number of periphery clusters
        center_std: Std dev for center clusters
        periphery_std: Std dev for periphery clusters
        random_ratio: Fraction of random cities
        seed: Random seed
    
    Returns:
        Coordinates [num_cities, 2]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    coords = []
    
    # Depot at origin
    coords.append(torch.tensor([0.0, 0.0]))
    
    num_remaining = num_cities - 1
    num_random = int(num_remaining * random_ratio)
    num_clustered = num_remaining - num_random
    
    # Generate center clusters
    center_centers = torch.rand(num_centers, 2)
    cities_per_center = num_clustered // (num_centers + num_periphery)
    
    for k in range(num_centers):
        n = cities_per_center
        cluster_coords = torch.randn(n, 2) * center_std + center_centers[k]
        cluster_coords = torch.clamp(cluster_coords, 0.0, 1.0)
        coords.append(cluster_coords)
    
    # Generate periphery clusters
    periphery_centers = torch.rand(num_periphery, 2)
    cities_per_periphery = (num_clustered - num_centers * cities_per_center) // num_periphery
    
    for k in range(num_periphery):
        n = cities_per_periphery
        cluster_coords = torch.randn(n, 2) * periphery_std + periphery_centers[k]
        cluster_coords = torch.clamp(cluster_coords, 0.0, 1.0)
        coords.append(cluster_coords)
    
    # Random cities
    random_coords = torch.rand(num_random, 2)
    coords.append(random_coords)
    
    coords = torch.cat(coords, dim=0)[:num_cities]
    
    return coords
```

**Статистичні властивості:**
- **Середня відстань між містами:** Залежить від розподілу кластерів
- **Щільність:** Вища в центрах, нижча на периферії
- **Реалістичність:** Краще моделює реальні міста

---

## 4. Обчислення матриці відстаней

### 4.1. Евклідова відстань (векторизована)

```python
def compute_distance_matrix(
    coords: torch.Tensor,
    metric: str = 'euclidean'
) -> torch.Tensor:
    """
    Compute pairwise distance matrix.
    
    Args:
        coords: City coordinates [N, 2]
        metric: Distance metric ('euclidean', 'manhattan', 'minkowski')
    
    Returns:
        Distance matrix [N, N]
    """
    if metric == 'euclidean':
        # ||x_i - x_j||_2
        # Expand: [N, 1, 2] - [1, N, 2] = [N, N, 2]
        coords_i = coords.unsqueeze(1)  # [N, 1, 2]
        coords_j = coords.unsqueeze(0)  # [1, N, 2]
        diff = coords_i - coords_j  # [N, N, 2]
        distances = torch.norm(diff, p=2, dim=-1)  # [N, N]
    
    elif metric == 'manhattan':
        coords_i = coords.unsqueeze(1)
        coords_j = coords.unsqueeze(0)
        diff = coords_i - coords_j
        distances = torch.norm(diff, p=1, dim=-1)
    
    elif metric == 'minkowski':
        # Use torch.cdist for general Minkowski distance
        distances = torch.cdist(coords, coords, p=2)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances
```

**Складність:** $O(N^2)$ — квадратична (необхідно для всіх пар).

**Пам'ять:** $O(N^2)$ — матриця відстаней.

**Оптимізація:** Використовуємо `torch.cdist` (оптимізована C++ реалізація):
```python
distances = torch.cdist(coords, coords, p=2)  # Euclidean
```

### 4.2. Додавання noise для реалістичності

```python
def add_distance_noise(
    distances: torch.Tensor,
    noise_std: float = 0.05,
    seed: int = None
) -> torch.Tensor:
    """
    Add multiplicative noise to distances (models road imperfections).
    
    Args:
        distances: Distance matrix [N, N]
        noise_std: Standard deviation of noise (as fraction)
        seed: Random seed
    
    Returns:
        Noisy distance matrix [N, N]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Multiplicative noise: d_new = d * (1 + ε), ε ~ N(0, σ²)
    noise = torch.randn_like(distances) * noise_std
    noisy_distances = distances * (1.0 + noise)
    
    # Ensure symmetry (distance matrix must be symmetric)
    noisy_distances = (noisy_distances + noisy_distances.T) / 2.0
    
    # Ensure non-negativity
    noisy_distances = torch.clamp(noisy_distances, min=0.0)
    
    return noisy_distances
```

**Математична формалізація:**
$$d_{ij}^{noisy} = d_{ij} \cdot (1 + \epsilon_{ij}), \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$$

**Обмеження:**
- Симетричність: $d_{ij} = d_{ji}$ (після додавання noise)
- Негативність: $d_{ij} \ge 0$ (clamp)

### 4.3. Матриця вартостей (Cost Matrix)

**Вартість може відрізнятися від відстані:**
- Вартість палива (залежить від рельєфу)
- Вартість часу (залежить від трафіку)
- Фіксовані витрати (плата за проїзд)

**Проста модель:**
$$c_{ij} = \alpha \cdot d_{ij} + \beta \cdot t_{ij} + \gamma$$

Де:
- $\alpha$ — вартість за одиницю відстані
- $\beta$ — вартість за одиницю часу
- $\gamma$ — фіксована вартість
- $t_{ij} = d_{ij} / v$ — час проїзду (при швидкості $v$)

**Для синтетичного бенчмарку (спрощено):**
$$c_{ij} = d_{ij}$$

(Вартість = відстань)

---

## 5. Генерація обмежень

### 5.1. Попит клієнтів (Demand)

**Розподіл попиту:**
$$q_i \sim \text{DiscreteUniform}(q_{min}, q_{max})$$

Або нормальний розподіл (з обрізанням):
$$q_i \sim \text{TruncatedNormal}(\mu_q, \sigma_q^2, q_{min}, q_{max})$$

**Реалізація:**
```python
def generate_demands(
    num_cities: int,
    demand_min: float = 1.0,
    demand_max: float = 10.0,
    depot_demand: float = 0.0,
    seed: int = None
) -> torch.Tensor:
    """
    Generate customer demands.
    
    Args:
        num_cities: Number of cities
        demand_min: Minimum demand
        demand_max: Maximum demand
        depot_demand: Demand at depot (usually 0)
        seed: Random seed
    
    Returns:
        Demands [num_cities]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    demands = torch.rand(num_cities) * (demand_max - demand_min) + demand_min
    
    # Depot has no demand
    demands[0] = depot_demand
    
    return demands
```

**Вантажопідйомність транспортного засобу:**
$$Q \ge \max_i q_i \quad \text{(можна обслуговувати найбільший клієнт)}$$
$$Q \ge \frac{\sum_i q_i}{K} \quad \text{(середній попит на транспортний засіб)}$$

Де $K$ — кількість транспортних засобів.

**Рекомендація:**
$$Q = \max\left(\max_i q_i, \frac{\sum_i q_i}{K} \cdot 1.2\right)$$

(20% запас для гнучкості).

### 5.2. Часові вікна (Time Windows)

**Генерація часових вікон:**
1. **Визначаємо час обслуговування:** $s_i$ (час на обслуговування клієнта $i$)
2. **Визначаємо час подорожі:** $t_{ij} = d_{ij} / v$ (відстань / швидкість)
3. **Генеруємо вікна:** $[e_i, l_i]$

**Алгоритм:**
1. Обчислюємо мінімальний час прибуття (через найкоротший шлях від депо)
2. Генеруємо earliest time: $e_i = t_{0i} + \text{random\_offset}$
3. Генеруємо latest time: $l_i = e_i + \text{window\_width}$

```python
def generate_time_windows(
    coords: torch.Tensor,
    distances: torch.Tensor,
    service_time: float = 10.0,
    speed: float = 1.0,
    window_width: float = 60.0,
    seed: int = None
) -> torch.Tensor:
    """
    Generate time windows for customers.
    
    Args:
        coords: City coordinates [N, 2]
        distances: Distance matrix [N, N]
        service_time: Time to serve a customer (minutes)
        speed: Average speed (distance units per minute)
        window_width: Width of time window (minutes)
        seed: Random seed
    
    Returns:
        Time windows [N, 2] where [:, 0] = earliest, [:, 1] = latest
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    N = coords.shape[0]
    time_windows = torch.zeros(N, 2)
    
    # Depot: always available
    time_windows[0] = torch.tensor([0.0, float('inf')])
    
    # Compute travel times
    travel_times = distances / speed  # [N, N]
    
    # Minimum arrival time from depot
    min_arrival = travel_times[0, :]  # [N]
    
    # Generate windows
    for i in range(1, N):
        # Earliest: minimum arrival + random offset
        earliest = min_arrival[i] + torch.rand(1).item() * 30.0
        
        # Latest: earliest + window width
        latest = earliest + window_width
        
        time_windows[i] = torch.tensor([earliest, latest])
    
    return time_windows
```

**Математична формалізація:**
$$e_i = t_{0i} + \epsilon_i, \quad \epsilon_i \sim \mathcal{U}(0, \Delta)$$
$$l_i = e_i + W$$

Де:
- $t_{0i}$ — час подорожі від депо до міста $i$
- $\Delta$ — максимальний offset (наприклад, 30 хвилин)
- $W$ — ширина вікна (наприклад, 60 хвилин)

### 5.3. Обмеження вантажопідйомності (Capacity)

**Перевірка feasibility:**
Маршрут $\pi = (\pi_1, \dots, \pi_k)$ є допустимим, якщо:
$$\sum_{i \in \pi} q_i \le Q$$

**Генерація вантажопідйомності:**
```python
def compute_vehicle_capacity(
    demands: torch.Tensor,
    num_vehicles: int,
    capacity_factor: float = 1.2
) -> float:
    """
    Compute vehicle capacity based on demands.
    
    Args:
        demands: Customer demands [N]
        num_vehicles: Number of vehicles
        capacity_factor: Multiplier for average demand
    
    Returns:
        Vehicle capacity Q
    """
    total_demand = demands.sum().item()
    avg_demand_per_vehicle = total_demand / num_vehicles
    max_demand = demands.max().item()
    
    capacity = max(max_demand, avg_demand_per_vehicle * capacity_factor)
    
    return capacity
```

---

## 6. Підготовка даних у форматі тензорів

### 6.1. Структура датасету

**Клас для екземпляра VRP:**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class VRPInstance:
    """Single VRP instance."""
    coords: torch.Tensor  # [N, 2] - city coordinates
    distances: torch.Tensor  # [N, N] - distance matrix
    costs: torch.Tensor  # [N, N] - cost matrix
    demands: torch.Tensor  # [N] - customer demands
    time_windows: Optional[torch.Tensor]  # [N, 2] - [earliest, latest]
    capacity: float  # Vehicle capacity
    num_vehicles: int  # Number of vehicles
    
    # Optional: ground truth solution
    optimal_route: Optional[torch.Tensor] = None
    optimal_cost: Optional[float] = None
```

### 6.2. Генератор датасету

```python
class VRPDatasetGenerator:
    """Generator for synthetic VRP datasets."""
    
    def __init__(
        self,
        coord_distribution: str = 'clustered',
        distance_metric: str = 'euclidean',
        add_noise: bool = True,
        noise_std: float = 0.05
    ):
        self.coord_distribution = coord_distribution
        self.distance_metric = distance_metric
        self.add_noise = add_noise
        self.noise_std = noise_std
    
    def generate_instance(
        self,
        num_cities: int,
        num_vehicles: int = None,
        seed: int = None
    ) -> VRPInstance:
        """
        Generate a single VRP instance.
        
        Args:
            num_cities: Number of cities (including depot)
            num_vehicles: Number of vehicles (default: sqrt(N))
            seed: Random seed
        
        Returns:
            VRPInstance
        """
        if num_vehicles is None:
            num_vehicles = int(np.sqrt(num_cities))
        
        # Generate coordinates
        if self.coord_distribution == 'uniform':
            coords = generate_uniform_cities(num_cities, seed)
        elif self.coord_distribution == 'clustered':
            coords = generate_clustered_cities(
                num_cities, 
                num_clusters=3, 
                seed=seed
            )
        elif self.coord_distribution == 'urban':
            coords = generate_urban_cities(num_cities, seed=seed)
        else:
            raise ValueError(f"Unknown distribution: {self.coord_distribution}")
        
        # Compute distances
        distances = compute_distance_matrix(coords, self.distance_metric)
        
        if self.add_noise:
            distances = add_distance_noise(distances, self.noise_std, seed)
        
        # Costs = distances (simplified)
        costs = distances.clone()
        
        # Generate demands
        demands = generate_demands(num_cities, seed=seed)
        
        # Generate time windows
        time_windows = generate_time_windows(
            coords, distances, seed=seed
        )
        
        # Compute capacity
        capacity = compute_vehicle_capacity(demands, num_vehicles)
        
        return VRPInstance(
            coords=coords,
            distances=distances,
            costs=costs,
            demands=demands,
            time_windows=time_windows,
            capacity=capacity,
            num_vehicles=num_vehicles
        )
    
    def generate_dataset(
        self,
        num_instances: int,
        num_cities_range: Tuple[int, int] = (20, 100),
        seed: int = None
    ) -> List[VRPInstance]:
        """
        Generate a dataset of VRP instances.
        
        Args:
            num_instances: Number of instances to generate
            num_cities_range: (min, max) range for number of cities
            seed: Random seed (for reproducibility)
        
        Returns:
            List of VRPInstance
        """
        if seed is not None:
            np.random.seed(seed)
        
        instances = []
        
        for i in range(num_instances):
            # Random number of cities in range
            num_cities = np.random.randint(
                num_cities_range[0], 
                num_cities_range[1] + 1
            )
            
            instance = self.generate_instance(
                num_cities=num_cities,
                seed=seed + i if seed is not None else None
            )
            
            instances.append(instance)
        
        return instances
```

### 6.3. Підготовка для навчання (DataLoader)

**Конвертація в тензори для батчування:**
```python
def collate_vrp_instances(
    instances: List[VRPInstance]
) -> Dict[str, torch.Tensor]:
    """
    Collate VRP instances into batched tensors.
    
    Args:
        instances: List of VRPInstance
    
    Returns:
        Dictionary with batched tensors
    """
    batch_size = len(instances)
    max_n = max(inst.n for inst in instances)
    
    # Initialize batched tensors
    coords = torch.zeros(batch_size, max_n, 2)
    distances = torch.zeros(batch_size, max_n, max_n)
    demands = torch.zeros(batch_size, max_n)
    time_windows = torch.zeros(batch_size, max_n, 2)
    masks = torch.zeros(batch_size, max_n, dtype=torch.bool)
    
    for i, inst in enumerate(instances):
        n = inst.coords.shape[0]
        
        # Pad to max_n
        coords[i, :n] = inst.coords
        distances[i, :n, :n] = inst.distances
        demands[i, :n] = inst.demands
        time_windows[i, :n] = inst.time_windows
        
        # Mask: True for valid cities, False for padding
        masks[i, :n] = True
    
    return {
        'coords': coords,
        'distances': distances,
        'demands': demands,
        'time_windows': time_windows,
        'masks': masks,
        'capacities': torch.tensor([inst.capacity for inst in instances]),
        'num_vehicles': torch.tensor([inst.num_vehicles for inst in instances])
    }
```

**Використання з PyTorch DataLoader:**
```python
from torch.utils.data import Dataset, DataLoader

class VRPDataset(Dataset):
    def __init__(self, instances: List[VRPInstance]):
        self.instances = instances
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]

# Create dataset
generator = VRPDatasetGenerator(coord_distribution='clustered')
instances = generator.generate_dataset(
    num_instances=1000,
    num_cities_range=(20, 50)
)
dataset = VRPDataset(instances)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_vrp_instances,
    shuffle=True
)
```

**Пам'ять для батчу:**
Для batch_size=32, max_n=50:
- coords: $32 \times 50 \times 2 \times 4$ байт = 12.8 KB
- distances: $32 \times 50 \times 50 \times 4$ байт = 320 KB
- Загалом: ~500 KB на батч (прийнятно)

---

## 7. Валідація та тестування

### 7.1. Перевірка feasibility

**Функція перевірки допустимості:**
```python
def is_feasible(
    instance: VRPInstance,
    route: torch.Tensor
) -> Tuple[bool, str]:
    """
    Check if route satisfies all constraints.
    
    Args:
        instance: VRP instance
        route: Route [route_length] - sequence of city indices
    
    Returns:
        (is_feasible, error_message)
    """
    # Check capacity
    route_demand = instance.demands[route].sum().item()
    if route_demand > instance.capacity:
        return False, f"Capacity violation: {route_demand} > {instance.capacity}"
    
    # Check time windows
    if instance.time_windows is not None:
        arrival_time = 0.0
        for i in range(len(route) - 1):
            from_city = route[i].item()
            to_city = route[i + 1].item()
            
            # Travel time
            travel_time = instance.distances[from_city, to_city].item()
            arrival_time += travel_time
            
            # Check time window
            earliest, latest = instance.time_windows[to_city]
            if arrival_time < earliest:
                arrival_time = earliest  # Wait
            elif arrival_time > latest:
                return False, f"Time window violation at city {to_city}"
            
            # Service time (simplified: 1 unit)
            arrival_time += 1.0
    
    return True, "Feasible"
```

### 7.2. Обчислення вартості маршруту

```python
def compute_route_cost(
    instance: VRPInstance,
    route: torch.Tensor
) -> float:
    """
    Compute total cost of route.
    
    Args:
        instance: VRP instance
        route: Route [route_length]
    
    Returns:
        Total cost
    """
    cost = 0.0
    
    for i in range(len(route) - 1):
        from_city = route[i].item()
        to_city = route[i + 1].item()
        cost += instance.costs[from_city, to_city].item()
    
    # Return to depot
    cost += instance.costs[route[-1].item(), 0].item()
    
    return cost
```

---

## 8. Engineering Challenge: AI-Resistant Assessment

### 8.1. Задача: Проектування генератора для production-like датасету

**Контекст:**
Потрібно створити генератор синтетичних датасетів, який максимально наближений до реальних даних доставки в місті.

**Вимоги:**
- **Реалістичність:** Географічний розподіл має відповідати реальним містам
- **Варіативність:** Підтримка різних розмірів задач ($N=10$ до $N=200$)
- **Контрольованість:** Параметри можна варіювати для аналізу
- **Ефективність:** Генерація 10,000 instances за $<5$ хвилин

**Реальні дані (аналіз):**
- Міста розподілені кластерами (центр + периферія)
- Відстані між містами: середнє 5-10 км, максимум 30 км
- Попит: більшість клієнтів 1-5 одиниць, рідко 10+
- Часові вікна: 80% клієнтів мають вікна 2-4 години, 20% — весь день

**Ваше завдання:**

1. **Спроектуйте архітектуру генератора:**
   - Які розподіли використати для координат?
   - Як моделювати дорожню мережу (не просто евклідова відстань)?
   - Як генерувати часові вікна з реалістичними паттернами?

2. **Обґрунтуйте вибір параметрів:**
   - Чому саме такі розподіли?
   - Як перевірити реалістичність (які метрики)?
   - Як забезпечити варіативність?

3. **Оцініть продуктивність:**
   - Складність генерації одного instance
   - Пам'ять для зберігання датасету
   - Час генерації 10,000 instances

**Критерії оцінки:**
- **Недостатньо:** "Використаємо uniform розподіл" (не реалістично)
- **Добре:** Детальний опис розподілів з обґрунтуванням
- **Відмінно:** Аналіз реалістичності через метрики, оптимізація продуктивності, порівняння з реальними даними

### 8.2. Референсне рішення (для викладача)

**Архітектура генератора:**

**1. Географічний розподіл:**
- **Кластерний з різними розмірами:** 2 великі кластери (центр, $\sigma=0.1$), 5 малих (периферія, $\sigma=0.05$)
- **Відстані між кластерами:** Мінімум 0.3 (щоб не перекривалися)
- **Масштабування:** Координати в $[0, 1]$, але з масштабом 30 км (1 одиниця = 30 км)

**2. Дорожня мережа:**
- **Гібридна метрика:** 70% евклідова + 30% Manhattan (моделює міську сітку)
- **Noise:** 5-10% варіація для моделювання неідеальності
- **Обмеження:** Мінімальна відстань 0.5 км (не можна бути занадто близько)

**3. Попит:**
- **Truncated Poisson:** $\lambda=3$, обрізаний до $[1, 10]$
- **80% клієнтів:** 1-5 одиниць
- **20% клієнтів:** 6-10 одиниць

**4. Часові вікна:**
- **Тип A (80%):** Вузькі вікна (2-4 години)
  - $e_i = t_{0i} + \mathcal{U}(0, 60)$ хвилин
  - $l_i = e_i + \mathcal{U}(120, 240)$ хвилин
- **Тип B (20%):** Широкі вікна (весь день)
  - $e_i = 0$
  - $l_i = \infty$ (або велике значення)

**Метрики реалістичності:**
1. **Середня відстань:** 5-10 км (перевірка)
2. **Щільність кластерів:** Вища в центрі (перевірка)
3. **Розподіл попиту:** 80/20 правило (перевірка)
4. **Розподіл часових вікон:** 80% вузькі, 20% широкі (перевірка)

**Продуктивність:**
- Генерація одного instance ($N=50$): ~10 ms
- 10,000 instances: ~100 секунд (1.7 хвилини) ✅
- Пам'ять: ~500 MB для 10,000 instances ✅

---

## 9. Джерела та Література

### 9.1. Генерація синтетичних датасетів
* **Ресурс:** [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) — Стандартний набір бенчмарків для TSP.
* **Ресурс:** [CVRPLIB](http://vrp.atd-lab.inf.puc-rio.br/index.php/en/) — Бенчмарки для VRP з різними обмеженнями.
* **Стаття:** *Uchoa, E., et al. (2017). "New benchmark instances for the Capacitated Vehicle Routing Problem".* [European Journal of Operational Research](https://www.sciencedirect.com/science/article/pii/S0377221716308770) — Методологія створення бенчмарків.

### 9.2. Статистичні розподіли та генерація
* **Книга:** *Devroye, L. "Non-Uniform Random Variate Generation".* [Springer](https://link.springer.com/book/10.1007/978-1-4613-8643-8) — Фундаментальна праця з генерації випадкових величин.
* **Ресурс:** [PyTorch Random Sampling](https://pytorch.org/docs/stable/torch.html#random-sampling) — Документація PyTorch для генерації випадкових чисел.

### 9.3. Підготовка даних для Deep Learning
* **Книга:** *Géron, A. "Hands-On Machine Learning".* [O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Розділ про підготовку даних.
* **Стаття:** *Kool, W., et al. (2019). "Attention, Learn to Solve Routing Problems!".* [ICLR 2019](https://arxiv.org/abs/1803.08475) — Приклад генерації датасету для TSP.

---

**Наступний крок:** Фінальне рішення та валідація ([10_solution_comparison.md](./10_solution_comparison.md)).

