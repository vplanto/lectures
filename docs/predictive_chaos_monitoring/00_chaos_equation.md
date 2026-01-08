---
title: "Логістичне відображення та Біфуркації"
layout: default
nav_order: 0
parent: "Вступ: Рівняння, що змінює погляд на світ"
---

# Логістичне відображення та Біфуркації

## Парадокс: Чому стабільність веде до колапсу

Уявіть сервер, який працює на 50% навантаження. Ви додаєте ще 10% — він працює на 60%. Логіка підказує: додайте ще 10% — отримаєте 70%. Але в реальності, коли система наближається до критичної точки, додавання навіть 1% може перетворити стабільну роботу на хаос. CPU спікується до 100%, черги розростаються експоненційно, і система падає.

Це не баг — це математика. І рівняння, яке описує цю поведінку, називається **логістичним відображенням**:

$$x_{n+1} = r \cdot x_n (1 - x_n)$$

Де:
- $x_n \in [0, 1]$ — стан системи на кроці $n$ (наприклад, нормалізоване навантаження CPU)
- $r \in [0, 4]$ — параметр контролю (коефіцієнт навантаження)
- $x_{n+1}$ — наступний стан

## Математичний фундамент

### Визначення та властивості

Логістичне відображення є дискретним динамічним системою, яка виникає з диференціального рівняння росту популяції з обмеженнями:

$$\frac{dx}{dt} = r x (1 - x)$$

Дискретизація методом Ейлера з кроком $h = 1$ дає нам ітераційну формулу. Функція $f_r(x) = r x (1 - x)$ називається **логістичною функцією**.

**Властивості:**
1. **Інваріантність інтервалу:** Якщо $x_0 \in [0, 1]$ та $r \in [0, 4]$, то $x_n \in [0, 1]$ для всіх $n$.
2. **Нерухомі точки:** Розв'язки $x^* = f_r(x^*)$:
   - $x_0^* = 0$ (тривіальна)
   - $x_1^* = 1 - \frac{1}{r}$ (існує при $r \geq 1$)

### Стабільність нерухомих точок

Лінеаризація навколо нерухомої точки $x^*$:

$$f_r(x^* + \epsilon) \approx f_r(x^*) + f_r'(x^*) \epsilon = x^* + f_r'(x^*) \epsilon$$

Де $f_r'(x) = r(1 - 2x)$ — похідна.

Нерухома точка **стабільна**, якщо $|f_r'(x^*)| < 1$:

- Для $x_0^* = 0$: $|f_r'(0)| = |r| < 1$ → стабільна при $r < 1$
- Для $x_1^* = 1 - \frac{1}{r}$: $|f_r'(x_1^*)| = |2 - r| < 1$ → стабільна при $1 < r < 3$

### Експонента Ляпунова: кількісна міра хаосу

Аналіз стабільності нерухомих точок дає нам якісну картину поведінки системи. Але для **кількісної оцінки хаосу** та визначення **горизонту передбачуваності** потрібна **експонента Ляпунова** $\lambda$.

**Визначення:** Для дискретної динамічної системи $x_{n+1} = f_r(x_n)$ експонента Ляпунова визначається як:

$$\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^{n-1} \ln |f_r'(x_i)|$$

Де $f_r'(x) = r(1 - 2x)$ — похідна логістичної функції.

**Фізичний зміст:** Експонента Ляпунова вимірює середню швидкість розбіжності близьких траєкторій:
- **$\lambda < 0$:** Траєкторії зближуються → система **стабільна** (періодична або збіжна)
- **$\lambda = 0$:** Нейтральна стабільність → **біфуркація**
- **$\lambda > 0$:** Траєкторії розбігаються експоненційно → система **хаотична**

**Горизонт передбачуваності:** Якщо початкова невизначеність становить $\delta_0$, то через $n$ кроків вона зросте приблизно до:

$$\delta_n \approx \delta_0 e^{\lambda n}$$

Горизонт передбачуваності $T$ (кількість кроків, після яких помилка стає порівнянною з розміром системи) визначається як:

$$T \approx \frac{1}{\lambda} \ln\left(\frac{1}{\delta_0}\right)$$

**Приклад:** Якщо $\lambda = 0.5$ (хаос) та $\delta_0 = 10^{-6}$, то $T \approx \frac{1}{0.5} \ln(10^6) \approx 28$ кроків. Це означає, що навіть при дуже точному знанні початкового стану, передбачення стає ненадійним вже через ~28 ітерацій.

**Для логістичного відображення:**
- При $r < 3$: $\lambda < 0$ (стабільна нерухома точка)
- При $3 < r < 3.569$: $\lambda < 0$ (періодичні орбіти)
- При $r > 3.569$: $\lambda > 0$ (хаос), причому $\lambda$ зростає з $r$

### Біфуркаційна діаграма

При $r = 3$ відбувається **перша біфуркація**: стабільна нерухома точка стає нестабільною, і система переходить до **2-циклу** (періодична орбіта з періодом 2).

При подальшому зростанні $r$:
- $r \approx 3.449$: 2-цикл → 4-цикл
- $r \approx 3.544$: 4-цикл → 8-цикл
- $r \approx 3.569$: 8-цикл → 16-цикл
- ...

Це явище називається **каскадом подвоєння періоду** (period-doubling cascade).

### Константа Фейгенбаума

Мітчел Фейгенбаум (1975) виявив універсальну закономірність:

$$\delta = \lim_{n \to \infty} \frac{r_n - r_{n-1}}{r_{n+1} - r_n} \approx 4.669201609...$$

Де $r_n$ — значення параметра, при якому відбувається $2^n$-біфуркація.

**Константа Фейгенбаума** $\delta$ є універсальною — вона з'являється в усіх системах, що демонструють каскад подвоєння періоду, незалежно від конкретної форми функції.

При $r > r_\infty \approx 3.5699$ система входить у **хаотичний режим**: поведінка стає аперіодичною та чутливою до початкових умов.

## Інженерна інтерпретація

### Аналогія: Laminar vs Turbulent Flow

У гідродинаміці перехід від ламінарної до турбулентної течії визначається числом Рейнольдса $Re$. Аналогічно, в IT-системах:

- **$r < 3$ (Ламінарний режим):** Система сходиться до стабільного стану. Метрики передбачувані, класичні пороги (CPU > 80%) працюють.
- **$3 < r < 3.569$ (Перехідний режим):** Періодичні коливання. Система "дихає" у ритмі, але пороги можуть спрацьовувати помилково.
- **$r > 3.569$ (Турбулентний/Хаотичний режим):** Аперіодична поведінка. Класичні пороги **не працюють**, бо:
  - Значення метрики може бути в нормі ($x_n \approx 0.3$), але наступний крок — спік ($x_{n+1} \approx 0.9$)
  - Малі зміни в навантаженні ($\Delta r \approx 0.01$) призводять до радикально різної поведінки

### Чому Thresholds не працюють у зоні турбулентності

Класичний моніторинг використовує статичні пороги:

```python
if cpu_usage > 0.8:
    send_alert("High CPU!")
```

У хаотичному режимі це еквівалентно спробі передбачити погоду, дивлячись лише на поточну температуру. Потрібен **динамічний аналіз траєкторії**, а не статичне значення.

## Реалізація на Python

### Генерація траєкторії логістичного відображення

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def logistic_map(r: float, x0: float, n_iter: int = 1000, 
                 transient: int = 100) -> np.ndarray:
    """
    Генерує траєкторію логістичного відображення.
    
    Parameters:
    -----------
    r : float
        Параметр контролю (0 <= r <= 4)
    x0 : float
        Початкова умова (0 <= x0 <= 1)
    n_iter : int
        Кількість ітерацій для збереження
    transient : int
        Кількість ітерацій для "прогріву" (відкидаються)
    
    Returns:
    --------
    trajectory : np.ndarray
        Масив значень x_n після transient ітерацій
    """
    x = x0
    trajectory = np.zeros(n_iter)
    
    # Пропускаємо transient для усунення впливу початкових умов
    for _ in range(transient):
        x = r * x * (1 - x)
    
    # Зберігаємо траєкторію
    for i in range(n_iter):
        x = r * x * (1 - x)
        trajectory[i] = x
    
    return trajectory

# Приклад: стабільна нерухома точка (r = 2.5)
trajectory_stable = logistic_map(r=2.5, x0=0.5, n_iter=100)
print(f"Стабільний режим (r=2.5): останні 5 значень = {trajectory_stable[-5:]}")

# Приклад: 2-цикл (r = 3.2)
trajectory_2cycle = logistic_map(r=3.2, x0=0.5, n_iter=100)
print(f"2-цикл (r=3.2): останні 5 значень = {trajectory_2cycle[-5:]}")

# Приклад: хаос (r = 3.8)
trajectory_chaos = logistic_map(r=3.8, x0=0.5, n_iter=100)
print(f"Хаос (r=3.8): останні 5 значень = {trajectory_chaos[-5:]}")
```

### Побудова біфуркаційної діаграми

```python
def bifurcation_diagram(r_min: float = 2.5, r_max: float = 4.0, 
                        r_steps: int = 1000, n_iter: int = 1000,
                        last_n: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Побудова біфуркаційної діаграми логістичного відображення.
    
    Parameters:
    -----------
    r_min, r_max : float
        Діапазон параметра r
    r_steps : int
        Кількість значень r для обчислення
    n_iter : int
        Кількість ітерацій для кожного r
    last_n : int
        Скільки останніх значень зберігати для побудови діаграми
    
    Returns:
    --------
    r_values : np.ndarray
        Масив значень параметра r
    x_values : np.ndarray
        Масив значень x (для побудови scatter plot)
    """
    r_values = np.linspace(r_min, r_max, r_steps)
    x_values = []
    r_plot = []
    
    for r in r_values:
        # Генеруємо траєкторію
        trajectory = logistic_map(r, x0=0.5, n_iter=n_iter, transient=200)
        # Зберігаємо останні last_n значень (після transient)
        x_values.extend(trajectory[-last_n:])
        r_plot.extend([r] * last_n)
    
    return np.array(r_plot), np.array(x_values)

# Побудова діаграми
r_plot, x_plot = bifurcation_diagram(r_min=2.5, r_max=4.0, r_steps=2000)

plt.figure(figsize=(12, 8))
plt.scatter(r_plot, x_plot, s=0.1, alpha=0.5, c='black')
plt.xlabel('Параметр $r$', fontsize=14)
plt.ylabel('$x_n$ (після збіжності)', fontsize=14)
plt.title('Біфуркаційна діаграма логістичного відображення', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xlim(2.5, 4.0)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('bifurcation_diagram.png', dpi=300)
plt.show()
```

### Обчислення константи Фейгенбаума

```python
def find_bifurcation_points(r_min: float = 3.0, r_max: float = 3.57,
                             precision: float = 1e-6) -> list:
    """
    Знаходить точки біфуркації методом бісекції.
    
    Returns:
    --------
    bifurcations : list
        Список значень r_n, при яких відбувається 2^n-біфуркація
    """
    bifurcations = []
    
    def period(x: np.ndarray, tol: float = 1e-6) -> int:
        """Визначає період орбіти."""
        for p in range(1, len(x) // 2):
            if np.allclose(x[-p:], x[-2*p:-p], atol=tol):
                return p
        return 0  # Хаос
    
    def bisect_bifurcation(r_low: float, r_high: float, target_period: int) -> float:
        """Бісекція для знаходження точки біфуркації."""
        while r_high - r_low > precision:
            r_mid = (r_low + r_high) / 2
            traj = logistic_map(r_mid, x0=0.5, n_iter=1000, transient=500)
            p = period(traj)
            
            if p < target_period:
                r_low = r_mid
            else:
                r_high = r_mid
        return (r_low + r_high) / 2
    
    # Знаходимо перші кілька біфуркацій
    current_period = 1
    r_current = 3.0
    
    for n in range(1, 6):  # До 16-циклу
        target_period = 2 ** n
        r_next = bisect_bifurcation(r_current, r_max, target_period)
        bifurcations.append(r_next)
        r_current = r_next
    
    return bifurcations

# Обчислення константи Фейгенбаума
bifurcations = find_bifurcation_points()
print("Точки біфуркації:")
for i, r in enumerate(bifurcations):
    print(f"  r_{i+1} (період {2**(i+1)}) = {r:.6f}")

if len(bifurcations) >= 3:
    ratios = []
    for i in range(len(bifurcations) - 2):
        ratio = (bifurcations[i+1] - bifurcations[i]) / (bifurcations[i+2] - bifurcations[i+1])
        ratios.append(ratio)
        print(f"  δ_{i+1} = {ratio:.6f}")
    
    print(f"\nНаближення константи Фейгенбаума: {np.mean(ratios):.6f}")
    print(f"Теоретичне значення: 4.669201609...")
```

### Аналіз чутливості до початкових умов

```python
def sensitivity_analysis(r: float = 3.8, x0_1: float = 0.5, 
                         x0_2: float = 0.500001, n_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Демонструє чутливість до початкових умов у хаотичному режимі.
    
    Returns:
    --------
    traj1, traj2 : np.ndarray
        Дві траєкторії з майже однаковими початковими умовами
    """
    traj1 = logistic_map(r, x0_1, n_iter=n_iter, transient=0)
    traj2 = logistic_map(r, x0_2, n_iter=n_iter, transient=0)
    
    return traj1, traj2

# Порівняння траєкторій
traj1, traj2 = sensitivity_analysis(r=3.8, x0_1=0.5, x0_2=0.500001, n_iter=30)

plt.figure(figsize=(12, 6))
plt.plot(traj1, 'b-', label='$x_0 = 0.5$', linewidth=2)
plt.plot(traj2, 'r--', label='$x_0 = 0.500001$', linewidth=2)
plt.xlabel('Ітерація $n$', fontsize=14)
plt.ylabel('$x_n$', fontsize=14)
plt.title('Чутливість до початкових умов (хаос, $r = 3.8$)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sensitivity_chaos.png', dpi=300)
plt.show()

# Обчислення розбіжності
divergence = np.abs(traj1 - traj2)
print(f"Початкова різниця: {divergence[0]:.2e}")
print(f"Різниця після 30 ітерацій: {divergence[-1]:.6f}")
print(f"Коефіцієнт зростання: {divergence[-1] / divergence[0]:.2e}")
```

### Обчислення експоненти Ляпунова

```python
def lyapunov_exponent(r: float, x0: float = 0.5, n_iter: int = 10000, 
                      transient: int = 1000) -> float:
    """
    Обчислює експоненту Ляпунова для логістичного відображення.
    
    Parameters:
    -----------
    r : float
        Параметр контролю (0 <= r <= 4)
    x0 : float
        Початкова умова (0 <= x0 <= 1)
    n_iter : int
        Кількість ітерацій для обчислення (більше = точніше)
    transient : int
        Кількість ітерацій для "прогріву" (відкидаються)
    
    Returns:
    --------
    lambda_val : float
        Експонента Ляпунова
    """
    x = x0
    
    # Пропускаємо transient для усунення впливу початкових умов
    for _ in range(transient):
        x = r * x * (1 - x)
    
    # Обчислюємо суму логарифмів похідної
    sum_log_derivative = 0.0
    
    for _ in range(n_iter):
        # Похідна: f_r'(x) = r(1 - 2x)
        derivative = abs(r * (1 - 2 * x))
        
        # Уникаємо логарифму від нуля (може статися при x = 0.5)
        if derivative > 1e-10:
            sum_log_derivative += np.log(derivative)
        
        # Оновлюємо x для наступної ітерації
        x = r * x * (1 - x)
    
    # Експонента Ляпунова
    lambda_val = sum_log_derivative / n_iter
    
    return lambda_val

# Обчислення експоненти Ляпунова для різних значень r
r_values = np.linspace(2.5, 4.0, 200)
lyapunov_values = []

for r in r_values:
    lambda_r = lyapunov_exponent(r, n_iter=5000, transient=500)
    lyapunov_values.append(lambda_r)

lyapunov_values = np.array(lyapunov_values)

# Побудова графіка експоненти Ляпунова
plt.figure(figsize=(12, 6))
plt.plot(r_values, lyapunov_values, 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1, label='$\lambda = 0$ (біфуркація)')
plt.axvline(x=3.569, color='g', linestyle='--', linewidth=1, alpha=0.7, 
            label='$r_\\infty \\approx 3.569$ (перехід до хаосу)')
plt.xlabel('Параметр $r$', fontsize=14)
plt.ylabel('Експонента Ляпунова $\\lambda$', fontsize=14)
plt.title('Експонента Ляпунова для логістичного відображення', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.xlim(2.5, 4.0)
plt.tight_layout()
plt.savefig('lyapunov_exponent.png', dpi=300)
plt.show()

# Аналіз для конкретних значень r
test_r_values = [2.5, 3.2, 3.5, 3.8, 4.0]
print("\nЕкспонента Ляпунова для різних режимів:")
print("-" * 50)
for r in test_r_values:
    lambda_r = lyapunov_exponent(r, n_iter=10000, transient=1000)
    regime = "Стабільний" if lambda_r < -0.1 else "Періодичний" if lambda_r < 0 else "Хаотичний"
    print(f"r = {r:.2f}: λ = {lambda_r:.6f} ({regime})")
```

### Горизонт передбачуваності

```python
def predictability_horizon(lambda_val: float, initial_uncertainty: float = 1e-6) -> float:
    """
    Обчислює горизонт передбачуваності на основі експоненти Ляпунова.
    
    Parameters:
    -----------
    lambda_val : float
        Експонента Ляпунова
    initial_uncertainty : float
        Початкова невизначеність (за замовчуванням 1e-6)
    
    Returns:
    --------
    T : float
        Горизонт передбачуваності (кількість кроків)
    """
    if lambda_val <= 0:
        return np.inf  # Система стабільна, передбачуваність необмежена
    
    # T ≈ (1/λ) * ln(1/δ₀)
    T = (1.0 / lambda_val) * np.log(1.0 / initial_uncertainty)
    return T

# Приклад: горизонт передбачуваності для хаотичного режиму
r_chaos = 3.8
lambda_chaos = lyapunov_exponent(r_chaos, n_iter=10000, transient=1000)
T_chaos = predictability_horizon(lambda_chaos, initial_uncertainty=1e-6)

print(f"\nХаотичний режим (r = {r_chaos}):")
print(f"  Експонента Ляпунова: λ = {lambda_chaos:.6f}")
print(f"  Горизонт передбачуваності: T ≈ {T_chaos:.1f} кроків")
print(f"  Інтерпретація: Навіть при точності 10⁻⁶, передбачення стає")
print(f"                  ненадійним вже через ~{int(T_chaos)} ітерацій")

# Порівняння з періодичним режимом
r_periodic = 3.2
lambda_periodic = lyapunov_exponent(r_periodic, n_iter=10000, transient=1000)
T_periodic = predictability_horizon(lambda_periodic, initial_uncertainty=1e-6)

print(f"\nПеріодичний режим (r = {r_periodic}):")
print(f"  Експонента Ляпунова: λ = {lambda_periodic:.6f}")
print(f"  Горизонт передбачуваності: T = {'∞' if T_periodic == np.inf else f'{T_periodic:.1f}'} кроків")
```

### Діаграма експоненти Ляпунова разом з біфуркаційною діаграмою

```python
# Комбінована візуалізація
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Біфуркаційна діаграма (зверху)
r_plot, x_plot = bifurcation_diagram(r_min=2.5, r_max=4.0, r_steps=2000)
ax1.scatter(r_plot, x_plot, s=0.1, alpha=0.5, c='black')
ax1.set_ylabel('$x_n$', fontsize=14)
ax1.set_title('Біфуркаційна діаграма та експонента Ляпунова', fontsize=16)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Експонента Ляпунова (знизу)
ax2.plot(r_values, lyapunov_values, 'b-', linewidth=2, label='$\\lambda(r)$')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=1, label='$\\lambda = 0$')
ax2.fill_between(r_values, 0, lyapunov_values, where=(lyapunov_values > 0), 
                 alpha=0.3, color='red', label='Хаос ($\\lambda > 0$)')
ax2.fill_between(r_values, 0, lyapunov_values, where=(lyapunov_values < 0), 
                 alpha=0.3, color='green', label='Стабільність ($\\lambda < 0$)')
ax2.set_xlabel('Параметр $r$', fontsize=14)
ax2.set_ylabel('$\\lambda$', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)
ax2.set_xlim(2.5, 4.0)

plt.tight_layout()
plt.savefig('bifurcation_lyapunov.png', dpi=300)
plt.show()
```

## Висновки та наступні кроки

Логістичне відображення демонструє фундаментальний принцип: **прості детерміновані правила можуть породжувати складну, передбачувану лише на коротких часових горизонтах поведінку**.

**Експонента Ляпунова** надає нам кількісний інструмент для оцінки стабільності та передбачуваності:
- **$\lambda < 0$:** Система стабільна, передбачення надійні на довгих горизонтах
- **$\lambda > 0$:** Система хаотична, горизонт передбачуваності обмежений: $T \approx \frac{1}{\lambda} \ln(1/\delta_0)$

Для SRE це означає:
1. **Статичні пороги недостатні** у зоні високого навантаження
2. **Потрібен динамічний аналіз траєкторії**, а не лише поточне значення метрики
3. **Малі зміни в навантаженні можуть призвести до радикальної зміни поведінки**
4. **Експонента Ляпунова дозволяє математично визначити горизонт передбачуваності** системи та адаптувати стратегії моніторингу відповідно до режиму роботи

**Практичне застосування:** У [практикумі з генерації синтетичного хаосу](07_synthetic_chaos_lab.md) ми демонструємо, як логістичне відображення використовується для створення тестових датасетів та дослідження меж передбачуваності LSTM моделей на хаотичних системах.

У наступних лекціях ми розглянемо:
- Як виміряти "пам'ять" системи (показник Херста)
- Як нейромережі (LSTM) можуть навчитися передбачати хаотичну поведінку
- Як побудувати предиктивний моніторинг на основі цих принципів

---

## Пов'язані теми

- **[Генерація датасету через Логістичне відображення](07_synthetic_chaos_lab.md)** — практичне застосування логістичного відображення для створення синтетичних датасетів та тестування LSTM моделей на хаотичних системах

---

## Додаткові матеріали

### Рекомендована література

1. Strogatz, S. H. (2014). *Nonlinear Dynamics and Chaos*. Westview Press.
2. Gleick, J. (1987). *Chaos: Making a New Science*. Viking.

### Вправи для самостійної роботи

1. **Завдання 1:** Модифікуйте функцію `logistic_map` для генерації траєкторії з додаванням білого шуму: $x_{n+1} = r x_n (1 - x_n) + \epsilon_n$, де $\epsilon_n \sim \mathcal{N}(0, \sigma^2)$. Дослідіть, як шум впливає на біфуркаційну діаграму.

2. **Завдання 2:** Реалізуйте обчислення **експоненти Ляпунова** $\lambda$ для логістичного відображення:
   $$\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^{n-1} \ln |f_r'(x_i)|$$
   Покажіть, що $\lambda > 0$ у хаотичному режимі.

3. **Завдання 3:** Створіть симуляцію навантаження CPU як логістичного відображення. Додайте механізм "автоскейлінгу", який зменшує $r$ при високому навантаженні. Порівняйте поведінку з класичним threshold-based алертингом.


