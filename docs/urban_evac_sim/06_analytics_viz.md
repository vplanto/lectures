# 06. Analytics & Observability Stack

**Scope:** KPI definition, Real-time Dashboarding, and Post-mortem Forensics.
**Input:** Time-series of Agent States $S(t)$ and Edge Weights $W(t)$.
**Output:** Analytical Report, Heatmaps, Anomaly Logs.

## 1. Система метрик (KPIs)

Ми розділяємо метрики на "дешеві" (Runtime), які не гальмують симуляцію, і "дорогі" (Forensic), які рахуємо після.

### 1.1. Global Survival Metrics (Runtime)
1.  **Total Clearance Time ($T_{99\%}$):**
    Час $t$, коли 99% активних агентів досягли зони $S_{sinks}$.
    * *Why 99%?* Щоб виключити статистичні викиди (агентів, що застрягли у дворах або "провалилися" крізь текстури карти).
2.  **Survival Rate ($S_{rate}$):**
    $$S_{rate} = \frac{N_{escaped}}{N_{total}} \cdot 100\%$$
    Агенти, що отримали статус `FAILED` (Fuel/Crash), вважаються втраченими.
3.  **Fuel Casualty Ratio ($FCR$):**
    Відсоток агентів, що зупинилися через вичерпання пального. Це головний індикатор ефективності роботи з чергами.

### 1.2. Network Health Metrics (Runtime Heuristics)
Пошук топологічних циклів (Deadlock Cycles) у реальному часі занадто дорогий ($O(V+E)$ на кожному кроці). Тому ми використовуємо **спрощені проксі-метрики**:

1.  **Gridlock Index ($G_{idx}$):**
    Відсоток дорожньої мережі, де рух зупинився.
    $$G_{idx}(t) = \frac{\sum_{e \in E} L_e \cdot \mathbb{I}(v_e(t) < v_{dead})}{L_{total}}$$
    * $v_{dead} \approx 0.5$ м/с.
    * Якщо $G_{idx} > 15\%$, система вважається заблокованою.

2.  **Mean System Velocity ($V_{sys}$):**
    Середньозважена швидкість усіх *активних* агентів. Якщо $V_{sys} \to 0$ при $N_{active} > 0$ — це системний колапс.

---

## 2. Real-time Dashboard (Runtime View)

Симуляція транслює стан для візуального дебагу.

### 2.1. Visualization Layers
1.  **Map View (Spatial):**
    * **Edges:** Колір залежить від $v_e$. Червоний = стоїмо.
    * **Cluster Alert:** Якщо група сусідніх ребер (Cluster) червона > 5 хв — підсвічувати як "Potentially Deadlocked Zone".
2.  **Agents:**
    * Не малювати 10,000 точок.
    * Малювати тільки *Rogue Agents* (червоним) та *Failed Agents* (чорні хрестики).

### 2.2. Architecture Pattern
* **Simulation Loop (Headless):** Працює на максимальній швидкості.
* **Rendering Loop:** Споживач (Consumer), читає буфер стану раз на 100мс (10 FPS). Це дозволяє симуляції не чекати на графіку.

---

## 3. Post-Mortem Analysis (Forensics)

Справжній аналіз причин колапсу відбувається **після** завершення симуляції, коли ми маємо повний лог подій і необмежений час CPU.

### 3.1. Deadlock Detection (Cycle Analysis)
Ми запускаємо алгоритм пошуку сильно зв'язних компонент (SCC) або циклів (наприклад, Tarjan's algo) на графі залежностей "хто кого блокував".

* **Input:** Лог блокувань (Agent A blocked Agent B at Node X).
* **Output:** Список ребер, що утворили цикл $e_A \to e_B \to e_C \to e_A$.
* **Insight:** Це ті самі "мертві петлі", які неможливо розв'язати без порушення ПДР (виїзду на зустрічну).

### 3.2. Failure Clusters
Географічний розподіл точок `FAILED`.
* Якщо кластер "померлих від палива" співпадає з в'їздом на міст — це структурна вразливість евакуаційного плану (Single Point of Failure).

### 3.3. Bottleneck Identification
Алгоритм розраховує "Вплив ребра": наскільки збільшився б $S_{rate}$, якби пропускна здатність ребра $e$ була нескінченною.
* Це дає рекомендацію для інженерів: "Розширте цей з'їзд, і врятуєте +500 людей".

---

## 4. Порівняльний аналіз (A/B/C Testing)

Кінцевий результат — графік залежності $T_{99}$ від стратегії.

**Очікувані криві:**
* **Regime A (Chaos):** Експоненційне зростання часу евакуації. Велика кількість Deadlock-циклів (виявлених на етапі Forensics).
* **Regime C (Optimum):** Лінійне зростання. Відсутність циклів завдяки превентивному контролю черг.

## 5. Validation Strategy: Sensitivity Analysis

**Problem Statement:**
Оскільки модель описує гіпотетичний сценарій (Evacuation under Fire), для неї не існує реальних історичних даних (Ground Truth) для калібрування параметрів (наприклад, ймовірності аварії $\beta$ або порогу паніки).

**Methodology:**
Замість спроб вгадати "істинні" значення коефіцієнтів, ми проводимо **Sensitivity Analysis** (Аналіз чутливості), щоб довести структурну стійкість моделі. Ми повинні показати, що висновки (Rank: $Regime_C > Regime_A$) зберігаються при варіації вхідних параметрів.

### 5.1. One-at-a-Time (OAT) Perturbation
Ми змінюємо один критичний параметр на $\pm 20\%$, фіксуючи інші, і дивимось на зміну цільової функції $T_{99}$.

| Parameter | Base Value | Range tested | Robustness Criteria |
| :--- | :--- | :--- | :--- |
| **Accident Prob ($\beta$)** | 0.05 / km | $0.01 \dots 0.10$ | $T_{99}$ changes linear, not exponential. |
| **Fuel Burn Rate** | 1.2 L/h | $0.8 \dots 2.0$ | $S_{rate}$ remains within $\pm 5\%$. |
| **Compliance ($\alpha$)** | 80% | $50\% \dots 100\%$ | Regime C must outperform A at any $\alpha > 30\%$. |

### 5.2. Search for Bifurcation Points
Ми шукаємо не точні цифри, а **фазові переходи**.
* *Приклад:* При якому значенні кількості *Rogue* агентів ($N_{rogue}$) система миттєво переходить зі стану "Slow Flow" у стан "Gridlock" (швидкість падає з 5 км/год до 0)?
* Якщо наша модель показує таку точку (Critical Threshold) — це ознака якісної симуляції складних систем, навіть якщо точне положення точки зміщене відносно реальності.

**Defense Argument:**
> "Ми не стверджуємо, що евакуація займе рівно 48 хвилин. Ми стверджуємо, що алгоритм 'Triage' є більш ефективним, ніж 'Chaos', при будь-якому рівні аварійності в діапазоні $[0.01; 0.1]$."

---

## 6. Technical Requirements for Viz

* **Static Plots:** `Matplotlib` / `Seaborn` (Spider charts для аналізу чутливості).
* **Interactive Map:** `Folium`.
* **Data Export:** `.csv` логи для обробки в Pandas/Excel.
