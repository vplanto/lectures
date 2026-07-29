---
title: "07 Tech Stack"
type: lecture
module: Module 7
prerequisites: module 6
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# 07. Implementation Stack & Optimization

**Scope:** Technology choices, Data structures, and Performance patterns.
**Constraint:** Python 3.10+ (Coursework limit), 10k+ Agents.

## 1. Core Stack

Ми не пишемо власний ігровий рушій на C++. Ми використовуємо стандартний науковий стек Python, але з розумінням його обмежень (Global Interpreter Lock, Object overhead).

### 1.1. Libraries
* **Graph & Topology:** `OSMnx` (завантаження), `NetworkX` (структура графа).
    * *Warning:* `NetworkX` написаний на чистому Python. Він підходить для побудови графа, але **повільний** для $10^4$ запитів шляху в секунду.
* **Math & State:** `NumPy` .
    * Векторні операції над станом агентів набагато швидші за цикли `for` по об'єктах.
* **Simulation Loop:** `Custom Loop` (While True).
    * Використання `SimPy` (Discrete Event) тут надлишкове, оскільки події (рух) відбуваються щокроку у всіх. Простий цикл дає кращий контроль.
* **Visuals:** `Folium` (звітність), `Matplotlib` (графіки).

---

## 2. Структури даних (Memory Layout)

Головна помилка новачків — створення "важких" об'єктів для кожного агента. 10,000 об'єктів Python — це гарантовані cache-misses.
Друга помилка — видалення агентів з масивів при "смерті". **Видалення рядка з NumPy масиву — це дорога операція копіювання пам'яті.**

### 2.1. Structure of Arrays (SoA)
Ми використовуємо фіксовані масиви. Агент ніколи не видаляється з пам'яті до кінця симуляції, він лише змінює статус.

```python
class Population:
    def __init__(self, n):
        # [EdgeID, Offset_meters]
        self.positions = np.zeros((n, 2), dtype=np.int32) 
        
        # [Speed, Fuel, Stress]
        self.state = np.zeros((n, 3), dtype=np.float32)
        
        # [Status] (0=Active, 1=Finished, 2=FAILED/DEAD)
        self.status = np.zeros(n, dtype=np.uint8)

    @property
    def active_mask(self):
        return self.status == 0

```

### 2.2. The "Zombie" Pattern (Handling Failures)

Коли у агента закінчується паливо, ми **не** видаляємо його з масивів.

1. **State Update:** Ми ставимо `status[i] = 2` (FAILED) та `speed[i] = 0`.
2. **Physics Logic:**
* У циклі оновлення позиції (Movement Step) ми використовуємо маску `active_mask`. Мертві агенти не рухаються.
* У циклі розрахунку щільності (Density Check) ми враховуємо **всіх** (`Active + Failed`).
* *Результат:* "Мертвий" агент залишається у графі як фізичний об'єкт, що займає місце (), створюючи затор, але перестає споживати CPU на пошук шляху.



### 2.3. Optimization Logic (Vectorized Update)

Оновлення фізики відбувається тільки для живих:

```python
def update_physics(self, dt):
    # 1. Identify who can move
    active = self.status == 0
    
    # 2. Update fuel (only for active, based on speed)
    # idle_consumption applied to all active, moving_consumption added if v > 0
    self.state[active, FUEL_IDX] -= (BASE_BURN + self.state[active, SPEED_IDX] * K_EFF) * dt
    
    # 3. Check for death
    new_dead = (self.state[:, FUEL_IDX] <= 0) & active
    self.status[new_dead] = 2  # Become Obstacle
    self.state[new_dead, SPEED_IDX] = 0 # Force Stop
    
    # 4. Move (only active)
    self.positions[active, OFFSET_IDX] += self.state[active, SPEED_IDX] * dt

```

Цей підхід гарантує коректну фізику заторів без накладних витрат на перебудову пам'яті.

---

## 3. High-Performance Computing (Vectorization Strategy)

Головний ризик Python-реалізації — це Overhead інтерпретатора. Цикл `for agent in agents:` для 10,000 об'єктів виконуватиметься ~50-100мс. Додайте сюди логіку графа — і ви отримаєте < 1 FPS.

**Вимога: Zero-Loop Physics**
Оновлення стану агентів має відбуватися виключно через **векторні операції NumPy** або **JIT-компіляцію**.

### 3.1. NumPy Broadcasting
Замість ітерації, ми оперуємо матрицями.
* *Wrong (Pythonic):*
  ```python
  for agent in population:
      if agent.speed < 0.1:
          agent.fuel -= 0.01

```

* *Right (Vectorized):*
```python
# Маска заторів (Boolean array)
stuck_mask = states[:, SPEED_IDX] < 0.1
# Віднімання палива тільки для тих, хто стоїть (in-place)
states[stuck_mask, FUEL_IDX] -= 0.01

```



Це прискорює розрахунки в ~50-100 разів.

### 3.2. Numba (JIT Compilation)

Для складних алгоритмів, які важко векторизувати (наприклад, Spatial Hash Lookup або специфічна логіка поведінки на перехресті), необхідно використовувати `@njit`.

```python
from numba import njit

@njit(fastmath=True)
def update_positions(positions, speeds, dt):
    # Цей цикл скомпілюється в машинний код (C-speed)
    for i in range(len(positions)):
        positions[i] += speeds[i] * dt

```

**Constraint:** Функції з `@njit` не можуть працювати з об'єктами Python (класами), тільки з масивами `numpy`. Це ще одна причина використовувати SoA (Structure of Arrays).

### 3.3. Graph Representation & A* Performance

Чітке розуміння структури графа є критичним для продуктивності алгоритму A*.

* **Чому не Матриця Суміжності (Adjacency Matrix)?**
  Студенти часто пропонують матрицю суміжності $N \times N$, оскільки це "класичний" спосіб з курсів алгоритмів. Для реальних міст це **помилка**. Урбаністичний граф ітеративно розріджений (sparse graph): кожне перехрестя (вузол) має в середньому 3-4 з'єднання, незалежно від розміру міста (degree $\approx 3.5$). Для графа з $10,000$ вузлів матриця матиме $10^8$ комірок, з яких $99.96\%$ будуть нулями. Це марна витрата пам'яті та кеш-промахи (cache misses) при кожному зверненні.
* **Правильний вибір: Списки Суміжності (Adjacency List) або Edge List (SoA)**
  Для алгоритму A* нам потрібна миттєва відповідь на запит "які сусіди у цього конкретного вузла". Тому ми використовуємо `Adjacency List` (або векторний еквівалент на базі індексів масивів у Numba).
* **NetworkX Performance:**
  `NetworkX` (написаний на чистому Python) занадто повільний для частих запитів `shortest_path` у реальному часі під напругою тисяч агентів.
  * *Рішення:* Використовувати кешування спільних (магістральних) шляхів, або одразу переходити на `igraph` / `graph-tool` (C++ бекенд) для важких розрахунків, особливо якщо сценарій вимагає частого перерахунку маршрутів для багатьох агентів одночасно (наприклад, у Regime C).

---

## 4. Архітектура коду (Project Layout)

```text
/src
  /core
    engine.py       # Main Loop (Physics -> Decisions -> Update)
    graph.py        # Graph wrapper (NetworkX + Weight Cache)
    spatial.py      # Spatial Hash Grid logic
  /models
    agents.py       # NumPy based agent state
    physics.py      # Fuel & Collision formulas
  /scenarios
    scenario_A.py   # Config for "Chaos"
    scenario_B.py   # Config for "Control"
  /utils
    osm_loader.py   # ETL logic
    metrics.py      # KPI calculators

```

---

## 5. Порядок розробки (Milestones)

Рекомендація рухатися ітераціями, щоб не застрягти:

1. **MVP 1 (The Dot):** Одне авто їде по одному порожньому ребру. Паливо витрачається.
2. **MVP 2 (The Graph):** Завантажити карту району. Авто їде з А в Б по .
3. **MVP 3 (The Crowd):** 100 авто. Реалізація черг (авто не проїжджають крізь авто).
4. **Alpha (The Chaos):** 1000 авто. Додавання Rogue behavior. Перші затори.
5. **Beta (The Control):** Введення режиму System Optimum. Збір метрик.
