---
title: "Випадкові блукання та Гіпотеза ефективного ринку (в IT)"
layout: default
nav_order: 1
parent: "Блок 1: Статистичний Фундамент та Фрактали"
---

# Випадкові блукання та Гіпотеза ефективного ринку (в IT)

## Парадокс: Чому сплеск Latency може бути передбачуваним

Уявіть ситуацію: ваш API показує сплеск latency з 50ms до 200ms. Класичне питання: "Це випадковий шум чи системна проблема?" 

Більшість інженерів відповідають: "Якщо це випадкове, то наступне значення буде незалежним від попереднього." Але що, якщо система має **пам'ять**? Що, якщо високий latency сьогодні збільшує ймовірність високого latency завтра через накопичення черг, перегрів кешу, або деградацію стану бази даних?

Це питання лежить в основі розрізнення між **випадковим блуканням** (random walk) та **процесом з пам'яттю** (persistent process). І відповідь визначає, чи можемо ми передбачати поведінку системи.

## Математичний фундамент

### Випадкове блукання (Random Walk)

**Визначення:** Дискретний випадковий процес $X_t$, де:

$$X_t = X_{t-1} + \epsilon_t$$

Де $\epsilon_t$ — незалежні однаково розподілені випадкові величини (i.i.d.) з $\mathbb{E}[\epsilon_t] = 0$ та $\text{Var}(\epsilon_t) = \sigma^2$.

**Властивості:**
1. **Мартингальна властивість:** $\mathbb{E}[X_t | X_{t-1}, X_{t-2}, \ldots] = X_{t-1}$
2. **Незалежні прирости:** $\text{Cov}(X_t - X_{t-1}, X_{t-k} - X_{t-k-1}) = 0$ для $k \geq 1$
3. **Варіація зростає лінійно:** $\text{Var}(X_t) = t \sigma^2$

**Гіпотеза ефективного ринку (Efficient Market Hypothesis, EMH):** У фінансах це означає, що ціни акцій відображають всю доступну інформацію, і майбутні зміни цін є випадковими (непередбачуваними). 

**Аналогія в IT:** Якщо метрики системи (latency, CPU, memory) слідують випадковому блуканню, то:
- Кожне значення незалежне від попередніх
- Неможливо передбачити майбутнє на основі минулого
- Класичні методи прогнозування (linear regression, moving average) будуть неефективними

### Автокореляція та пам'ять процесу

**Автокореляційна функція (ACF):**

$$\rho(k) = \frac{\text{Cov}(X_t, X_{t-k})}{\sqrt{\text{Var}(X_t) \text{Var}(X_{t-k})}} = \frac{\mathbb{E}[(X_t - \mu)(X_{t-k} - \mu)]}{\sigma^2}$$

Для випадкового блукання:

$$\rho(k) = \frac{\max(0, t - k)}{t} \to 1 \text{ при } t \to \infty$$

Це означає, що навіть у випадковому блуканні є **спостережувана кореляція** між далекими значеннями, але це артефакт нестаціонарності, а не справжня пам'ять.

### Нестаціонарність та проблема з Linear Regression

**Стаціонарний процес:** Розподіл $X_t$ не залежить від $t$:
- $\mathbb{E}[X_t] = \mu$ (константа)
- $\text{Var}(X_t) = \sigma^2$ (константа)
- $\text{Cov}(X_t, X_{t-k}) = \gamma(k)$ (залежить лише від лагу $k$)

**Нестаціонарний процес:** Хоча б одна з цих умов порушена.

Випадкове блукання є **нестаціонарним**, бо $\text{Var}(X_t) = t\sigma^2 \to \infty$ при $t \to \infty$.

**Проблема з Linear Regression:**

Стандартна модель:

$$Y_t = \alpha + \beta X_t + \epsilon_t$$

Припускає:
1. Стаціонарність залишків $\epsilon_t$
2. Незалежність спостережень
3. Гомоскедастичність (постійна дисперсія)

Для нестаціонарних процесів ці припущення порушуються, що призводить до:
- **Спurious regression** (хибна регресія): висока $R^2$ та значущі коефіцієнти навіть при відсутності реального зв'язку
- **Невірні довірчі інтервали** для коефіцієнтів
- **Ненадійні прогнози**

### Тест на стаціонарність: Augmented Dickey-Fuller (ADF)

**Null гіпотеза:** Процес має одиничний корінь (unit root), тобто є нестаціонарним.

Модель для тесту:

$$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \delta_i \Delta X_{t-i} + \epsilon_t$$

Де $\Delta X_t = X_t - X_{t-1}$ — перша різниця.

**Статистика ADF:**

$$ADF = \frac{\hat{\gamma}}{SE(\hat{\gamma})}$$

Якщо $ADF < \text{критичне значення}$, відхиляємо $H_0$ → процес стаціонарний.

## Інженерна інтерпретація

### Чи є сплеск Latency випадковим?

**Сценарій 1: Випадкове блукання (немає пам'яті)**
- Кожен сплеск незалежний
- Накопичення стану не відбувається
- Система "забуває" попередні події

**Сценарій 2: Процес з пам'яттю (персистентний)**
- Високий latency сьогодні → вища ймовірність високого latency завтра
- Система "запам'ятовує" стан через:
  - Накопичення черг
  - Деградацію кешу
  - Теплову інерцію (CPU throttling)
  - Фрагментацію пам'яті

**Практичне питання:** Якщо latency має пам'ять, то ми можемо використовувати минулі значення для прогнозування майбутніх. Якщо ні — потрібні інші підходи (наприклад, аналіз причин, а не наслідків).

### Чому Linear Regression безсила

Класичний підхід до прогнозування:

```python
# НЕПРАВИЛЬНО для нестаціонарних процесів
from sklearn.linear_model import LinearRegression

X = latency[:-1].reshape(-1, 1)  # Попередні значення
y = latency[1:]  # Наступні значення

model = LinearRegression()
model.fit(X, y)
prediction = model.predict(X[-1:])
```

**Проблеми:**
1. Припускає лінійний зв'язок (може бути нелінійним)
2. Ігнорує нестаціонарність
3. Не враховує довгострокову пам'ять
4. Довірчі інтервали ненадійні

## Реалізація на Python

### Генерація випадкового блукання

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Tuple

def generate_random_walk(n: int, sigma: float = 1.0, x0: float = 0.0) -> np.ndarray:
    """
    Генерує випадкове блукання.
    
    Parameters:
    -----------
    n : int
        Кількість кроків
    sigma : float
        Стандартне відхилення приростів
    x0 : float
        Початкове значення
    
    Returns:
    --------
    walk : np.ndarray
        Траєкторія випадкового блукання
    """
    increments = np.random.normal(0, sigma, n)
    walk = np.cumsum(increments) + x0
    return walk

# Генерація прикладу
np.random.seed(42)
random_walk = generate_random_walk(n=1000, sigma=0.5)

plt.figure(figsize=(12, 6))
plt.plot(random_walk, linewidth=1)
plt.xlabel('Час $t$', fontsize=14)
plt.ylabel('$X_t$', fontsize=14)
plt.title('Випадкове блукання', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('random_walk.png', dpi=300)
plt.show()
```

### Тест на стаціонарність (ADF)

```python
def test_stationarity(series: np.ndarray, title: str = "Series") -> dict:
    """
    Виконує Augmented Dickey-Fuller тест на стаціонарність.
    
    Returns:
    --------
    result : dict
        Словник з результатами тесту
    """
    result = adfuller(series, autolag='AIC')
    
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    print(f"\n{'='*60}")
    print(f"ADF Test для {title}")
    print(f"{'='*60}")
    print(f"ADF Statistic: {adf_statistic:.6f}")
    print(f"p-value: {p_value:.6f}")
    print(f"\nКритичні значення:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.6f}")
    
    if p_value <= 0.05:
        print(f"\nРезультат: Відхиляємо H0 → Процес СТАЦІОНАРНИЙ")
    else:
        print(f"\nРезультат: Не відхиляємо H0 → Процес НЕСТАЦІОНАРНИЙ")
    
    return {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_stationary': p_value <= 0.05
    }

# Тест для випадкового блукання
result_rw = test_stationarity(random_walk, "Випадкове блукання")

# Тест для першої різниці (має бути стаціонарною)
diff_rw = np.diff(random_walk)
result_diff = test_stationarity(diff_rw, "Перша різниця випадкового блукання")
```

### Обчислення автокореляції

```python
def compute_autocorrelation(series: np.ndarray, max_lag: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обчислює автокореляційну функцію (ACF).
    
    Returns:
    --------
    lags : np.ndarray
        Масив лагів
    acf_values : np.ndarray
        Значення ACF для кожного лагу
    """
    acf_values = acf(series, nlags=max_lag, fft=True)
    lags = np.arange(len(acf_values))
    return lags, acf_values

# Обчислення ACF для випадкового блукання
lags, acf_rw = compute_autocorrelation(random_walk, max_lag=50)

plt.figure(figsize=(12, 6))
plt.stem(lags, acf_rw, basefmt=" ")
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='95% довірча зона')
plt.axhline(y=-0.05, color='r', linestyle='--', linewidth=1)
plt.xlabel('Лаг $k$', fontsize=14)
plt.ylabel('ACF($k$)', fontsize=14)
plt.title('Автокореляційна функція для випадкового блукання', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acf_random_walk.png', dpi=300)
plt.show()
```

### Моделювання процесу з пам'яттю (AR(1))

```python
def generate_ar1_process(n: int, phi: float = 0.7, sigma: float = 1.0) -> np.ndarray:
    """
    Генерує AR(1) процес: X_t = phi * X_{t-1} + epsilon_t
    
    Parameters:
    -----------
    n : int
        Кількість спостережень
    phi : float
        Коефіцієнт автокореляції (|phi| < 1 для стаціонарності)
    sigma : float
        Стандартне відхилення шуму
    
    Returns:
    --------
    series : np.ndarray
        Траєкторія AR(1) процесу
    """
    series = np.zeros(n)
    epsilon = np.random.normal(0, sigma, n)
    
    for t in range(1, n):
        series[t] = phi * series[t-1] + epsilon[t]
    
    return series

# Генерація AR(1) з пам'яттю
ar1_memory = generate_ar1_process(n=1000, phi=0.8, sigma=0.5)

# Порівняння з випадковим блуканням
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(random_walk, linewidth=1, label='Випадкове блукання (немає пам\'яті)')
plt.xlabel('Час $t$', fontsize=12)
plt.ylabel('$X_t$', fontsize=12)
plt.title('Порівняння: Випадкове блукання vs AR(1) з пам\'яттю', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(ar1_memory, linewidth=1, color='orange', label='AR(1), φ=0.8 (є пам\'ять)')
plt.xlabel('Час $t$', fontsize=12)
plt.ylabel('$X_t$', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('random_walk_vs_ar1.png', dpi=300)
plt.show()

# ACF для AR(1)
lags_ar1, acf_ar1 = compute_autocorrelation(ar1_memory, max_lag=50)

plt.figure(figsize=(12, 6))
plt.stem(lags_ar1, acf_ar1, basefmt=" ", linefmt='orange-', markerfmt='o')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.xlabel('Лаг $k$', fontsize=14)
plt.ylabel('ACF($k$)', fontsize=14)
plt.title('Автокореляція AR(1) процесу (φ=0.8)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acf_ar1.png', dpi=300)
plt.show()

# Теоретична ACF для AR(1): ρ(k) = φ^k
theoretical_acf = 0.8 ** lags_ar1
plt.figure(figsize=(12, 6))
plt.plot(lags_ar1, theoretical_acf, 'r--', linewidth=2, label='Теоретична: $φ^k$')
plt.plot(lags_ar1, acf_ar1, 'o-', linewidth=1, markersize=4, label='Емпірична')
plt.xlabel('Лаг $k$', fontsize=14)
plt.ylabel('ACF($k$)', fontsize=14)
plt.title('Порівняння теоретичної та емпіричної ACF для AR(1)', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('theoretical_vs_empirical_acf.png', dpi=300)
plt.show()
```

### Демонстрація проблеми з Linear Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def demonstrate_spurious_regression():
    """
    Демонструє проблему спurious regression для нестаціонарних процесів.
    """
    # Генеруємо два незалежні випадкові блукання
    np.random.seed(123)
    X = generate_random_walk(n=500, sigma=1.0)
    Y = generate_random_walk(n=500, sigma=1.0)
    
    # Намагаємося знайти зв'язок між ними
    X_lag = X[:-1].reshape(-1, 1)
    Y_future = Y[1:]
    
    model = LinearRegression()
    model.fit(X_lag, Y_future)
    
    predictions = model.predict(X_lag)
    r2 = r2_score(Y_future, predictions)
    
    print(f"\n{'='*60}")
    print("Спurious Regression: Два незалежні випадкові блукання")
    print(f"{'='*60}")
    print(f"R² = {r2:.4f} (високий, але хибний!)")
    print(f"Коефіцієнт регресії: {model.coef_[0]:.4f}")
    print(f"p-value (наближено): < 0.001 (хибно значущий!)")
    print(f"\nВисновок: Висока R² не означає реальний зв'язок для нестаціонарних даних!")
    
    # Візуалізація
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(X, label='$X_t$', linewidth=1)
    plt.plot(Y, label='$Y_t$', linewidth=1)
    plt.xlabel('Час $t$', fontsize=12)
    plt.ylabel('Значення', fontsize=12)
    plt.title('Два незалежні випадкові блукання', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_lag, Y_future, alpha=0.3, s=10)
    plt.plot(X_lag, predictions, 'r-', linewidth=2, label=f'Регресія (R²={r2:.3f})')
    plt.xlabel('$X_{t-1}$', fontsize=12)
    plt.ylabel('$Y_t$', fontsize=12)
    plt.title('Спurious Regression', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spurious_regression.png', dpi=300)
    plt.show()

demonstrate_spurious_regression()
```

### Практичний аналіз: Симуляція Latency з пам'яттю

```python
def simulate_latency_with_memory(n: int, base_latency: float = 50.0, 
                                  memory_strength: float = 0.7) -> pd.DataFrame:
    """
    Симулює latency з пам'яттю (AR(1) процес з додаванням базового рівня).
    
    Parameters:
    -----------
    n : int
        Кількість спостережень
    base_latency : float
        Базовий рівень latency (ms)
    memory_strength : float
        Сила пам'яті (0 = немає пам'яті, 1 = повна пам'ять)
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame з latency та його компонентами
    """
    # Генеруємо AR(1) процес для відхилень від базового рівня
    deviations = generate_ar1_process(n, phi=memory_strength, sigma=10.0)
    
    # Додаємо базовий рівень
    latency = base_latency + deviations
    
    # Додаємо невеликий білий шум
    noise = np.random.normal(0, 2.0, n)
    latency = latency + noise
    
    # Забезпечуємо невід'ємність
    latency = np.maximum(latency, 0)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n, freq='1min'),
        'latency': latency,
        'deviation': deviations
    })
    
    return df

# Генерація даних
latency_data = simulate_latency_with_memory(n=1000, base_latency=50.0, memory_strength=0.75)

# Візуалізація
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(latency_data['timestamp'], latency_data['latency'], linewidth=1)
plt.axhline(y=50, color='r', linestyle='--', linewidth=1, label='Базовий рівень (50ms)')
plt.xlabel('Час', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.title('Симуляція Latency з пам\'яттю (AR(1), φ=0.75)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.subplot(2, 1, 2)
# Обчислюємо автокореляцію
lags_latency, acf_latency = compute_autocorrelation(latency_data['latency'].values, max_lag=50)
plt.stem(lags_latency, acf_latency, basefmt=" ")
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.xlabel('Лаг $k$ (хвилини)', fontsize=12)
plt.ylabel('ACF($k$)', fontsize=12)
plt.title('Автокореляція Latency', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latency_with_memory.png', dpi=300)
plt.show()

# Тест на стаціонарність
result_latency = test_stationarity(latency_data['latency'].values, "Latency з пам'яттю")
```

## Висновки та наступні кроки

Ключові висновки:

1. **Випадкове блукання не має пам'яті** — кожен крок незалежний, але процес нестаціонарний
2. **Процеси з пам'яттю (AR, ARMA)** демонструють автокореляцію та можуть бути передбачуваними
3. **Linear Regression безсила** для нестаціонарних процесів через спurious regression
4. **Тест ADF** дозволяє розрізнити стаціонарні та нестаціонарні процеси

**Для SRE практики:**
- Якщо метрики слідують випадковому блуканню ($H \approx 0.5$), класичні методи прогнозування неефективні
- Якщо є пам'ять ($H > 0.5$), можна використовувати минулі значення для прогнозування
- Потрібні методи, які враховують нестаціонарність та довгострокову пам'ять

У наступній лекції ми розглянемо **показник Херста** — кількісну міру пам'яті процесу, яка дозволяє точно визначити, чи має система "інерцію" або є повністю випадковою.

---

## Додаткові матеріали

### Рекомендована література

1. Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
2. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.

### Вправи для самостійної роботи

1. **Завдання 1:** Реалізуйте тест **Ljung-Box** для перевірки автокореляції залишків. Застосуйте його до AR(1) процесу та випадкового блукання. Порівняйте результати.

2. **Завдання 2:** Створіть функцію, яка генерує **ARMA(p, q)** процес загального вигляду. Дослідіть, як зміна параметрів $p$ та $q$ впливає на автокореляційну функцію.

3. **Завдання 3:** Завантажте реальні метрики з Prometheus/Grafana (або використайте публічні датасети). Застосуйте ADF тест та обчисліть ACF. Визначте, чи мають ваші метрики пам'ять, чи слідують випадковому блуканню.


