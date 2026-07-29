---
title: "02 Hurst Exponent Fractals"
type: lecture
module: Module 2
prerequisites: module 1
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# R/S Аналіз та Показник Херста

## Парадокс: Як виміряти "пам'ять" у випадкових даних

Уявіть два графіки CPU utilization:
- **Графік A:** Виглядає як білий шум, значення стрибають випадково
- **Графік B:** Показує "тренди" — якщо CPU високий зараз, він має тенденцію залишатися високим

Інтуїтивно зрозуміло, що Графік B має **пам'ять** — минуле впливає на майбутнє. Але як це виміряти кількісно?

У 1951 році Гарольд Едвін Херст, досліджуючи рівень води в Нілі, розробив метод **R/S аналізу** (Rescaled Range Analysis), який дозволяє обчислити **показник Херста** $H$ — міру довгострокової залежності в часовому ряді.

## Математичний фундамент

### Визначення показника Херста

**Показник Херста** $H \in [0, 1]$ характеризує поведінку часового ряду:

$$R/S \sim (n/2)^H$$

Де:
- $R$ — діапазон (range) накопичених відхилень
- $S$ — стандартне відхилення
- $n$ — довжина підпослідовності

**Інтерпретація:**
- **$H = 0.5$:** Броунівський рух (випадкове блукання). Немає пам'яті, майбутнє незалежне від минулого.
- **$0 < H < 0.5$:** Антиперсистентний процес. Система має тенденцію "повертатися" до середнього (mean-reverting).
- **$0.5 < H \leq 1$:** Персистентний процес. Система має "пам'ять" — тренди мають тенденцію продовжуватися.

### R/S Аналіз: Алгоритм

Для часового ряду $X = \{X_1, X_2, \ldots, X_N\}$:

**Крок 1:** Розбиття на підпослідовності довжини $n$:
- $n \in \{n_1, n_2, \ldots, n_k\}$, де $n_1 < n_2 < \ldots < n_k$
- Типові значення: $n_i = 2^i$ або $n_i = \lfloor N/2^i \rfloor$

**Крок 2:** Для кожної підпослідовності $X^{(j)} = \{X_{j+1}, \ldots, X_{j+n}\}$:

1. **Обчислення середнього:**
   $$\bar{X}^{(j)} = \frac{1}{n} \sum_{i=1}^{n} X_{j+i}$$

2. **Накопичені відхилення:**
   $$Y_k^{(j)} = \sum_{i=1}^{k} (X_{j+i} - \bar{X}^{(j)}), \quad k = 1, 2, \ldots, n$$

3. **Діапазон (Range):**
   $$R^{(j)}(n) = \max_{1 \leq k \leq n} Y_k^{(j)} - \min_{1 \leq k \leq n} Y_k^{(j)}$$

4. **Стандартне відхилення:**
   $$S^{(j)}(n) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (X_{j+i} - \bar{X}^{(j)})^2}$$

5. **Нормалізований діапазон:**
   $$(R/S)^{(j)}(n) = \frac{R^{(j)}(n)}{S^{(j)}(n)}$$

**Крок 3:** Усереднення по всіх підпослідовностях:
$$\overline{R/S}(n) = \frac{1}{m(n)} \sum_{j=1}^{m(n)} (R/S)^{(j)}(n)$$

Де $m(n)$ — кількість підпослідовностей довжини $n$.

**Крок 4:** Лінеаризація та оцінка $H$:
$$\log(\overline{R/S}(n)) = H \log(n) + \log(C)$$

Оцінка $H$ виконується через лінійну регресію $\log(\overline{R/S}(n))$ на $\log(n)$.

### Фрактальна розмірність

**Зв'язок з фрактальною розмірністю:**

$$D = 2 - H$$

Де $D$ — фрактальна розмірність часового ряду.

**Інтерпретація:**
- $H = 0.5 \rightarrow D = 1.5$: Броунівський рух (класична фрактальна розмірність)
- $H \to 1 \rightarrow D \to 1$: Гладкий, детермінований процес
- $H \to 0 \rightarrow D \to 2$: Високо фрактальний, сильно коливальний

### Теоретичні властивості

**Для фракційного броунівського руху (fBm):**

Фракційний броунівський рух $B_H(t)$ з показником Херста $H$ має властивість:

$$\mathbb{E}[(B_H(t) - B_H(s))^2] = |t - s|^{2H}$$

**Автоковаріаційна функція:**

$$\text{Cov}(B_H(t), B_H(s)) = \frac{1}{2}(|t|^{2H} + |s|^{2H} - |t-s|^{2H})$$

Для $H > 0.5$ це дає позитивну автокореляцію на довгих лагах.

## Інженерна інтерпретація

### Практичне значення показника Херста для SRE

**$H \approx 0.5$ (Броунівський рух):**
- Метрика поводиться як випадковий шум
- Немає довгострокової пам'яті
- Класичні методи прогнозування неефективні
- **Рекомендація:** Фокусуватися на аналізі причин, а не наслідків

**$0.5 < H \leq 0.7$ (Слабка персистентність):**
- Помірна пам'ять, короткострокові тренди
- Можна використовувати прості методи (moving average, exponential smoothing)
- **Рекомендація:** Моніторинг з короткими вікнами

**$0.7 < H \leq 1.0$ (Сильна персистентність):**
- Сильна пам'ять, довгострокові тренди
- Система має "інерцію" — стан зберігається
- **Рекомендація:** Використання LSTM та інших методів, що враховують довгу історію

### Фільтрація метрик за показником Херста

**Стратегія:** Ігнорувати метрики з $H \approx 0.5$ (немає передбачуваності) та фокусуватися на метриках з $H > 0.7$ (сильна пам'ять, можна передбачати).

**Переваги:**
1. Зменшення шуму в системі алертингу
2. Фокус на метриках, які дійсно можна передбачити
3. Економія обчислювальних ресурсів (не навчаємо моделі на випадковому шумі)

## Реалізація на Python

### Реалізація R/S аналізу

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

def rs_analysis(series: np.ndarray, min_window: int = 10, 
                max_window: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Виконує R/S аналіз для обчислення показника Херста.
    
    Parameters:
    -----------
    series : np.ndarray
        Часовий ряд
    min_window : int
        Мінімальна довжина вікна
    max_window : int
        Максимальна довжина вікна (None = половина довжини ряду)
    
    Returns:
    --------
    log_n : np.ndarray
        Логарифми довжин вікон
    log_rs : np.ndarray
        Логарифми усереднених R/S значень
    """
    N = len(series)
    if max_window is None:
        max_window = N // 2
    
    # Генерація послідовності довжин вікон
    # Використовуємо геометричну прогресію
    windows = []
    n = min_window
    while n <= max_window:
        windows.append(n)
        n = int(n * 1.2)  # Збільшуємо на 20%
    
    log_n = []
    log_rs = []
    
    for n in windows:
        # Кількість підпослідовностей
        m = N // n
        if m < 2:
            continue
        
        rs_values = []
        
        for j in range(m):
            # Витягуємо підпослідовність
            subseries = series[j*n:(j+1)*n]
            
            if len(subseries) < n:
                continue
            
            # Середнє
            mean_sub = np.mean(subseries)
            
            # Накопичені відхилення
            deviations = subseries - mean_sub
            cumsum_deviations = np.cumsum(deviations)
            
            # Діапазон (Range)
            R = np.max(cumsum_deviations) - np.min(cumsum_deviations)
            
            # Стандартне відхилення
            S = np.std(subseries, ddof=0)
            
            # Уникаємо ділення на нуль
            if S > 1e-10:
                rs_values.append(R / S)
        
        if len(rs_values) > 0:
            # Усереднення
            mean_rs = np.mean(rs_values)
            log_n.append(np.log(n))
            log_rs.append(np.log(mean_rs))
    
    return np.array(log_n), np.array(log_rs)

def estimate_hurst(series: np.ndarray, min_window: int = 10, 
                   max_window: int = None, plot: bool = True) -> Tuple[float, dict]:
    """
    Оцінює показник Херста через R/S аналіз.
    
    Returns:
    --------
    H : float
        Оцінка показника Херста
    results : dict
        Словник з детальними результатами
    """
    log_n, log_rs = rs_analysis(series, min_window, max_window)
    
    if len(log_n) < 2:
        raise ValueError("Недостатньо точок для оцінки H")
    
    # Лінійна регресія: log(R/S) = H * log(n) + C
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_n, log_rs)
    
    H = slope
    
    results = {
        'H': H,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err,
        'log_n': log_n,
        'log_rs': log_rs,
        'fractal_dimension': 2 - H
    }
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(log_n, log_rs, alpha=0.6, s=50, label='Дані')
        
        # Лінія регресії
        log_n_fit = np.linspace(log_n.min(), log_n.max(), 100)
        log_rs_fit = H * log_n_fit + intercept
        plt.plot(log_n_fit, log_rs_fit, 'r-', linewidth=2, 
                label=f'Регресія: H = {H:.3f}')
        
        plt.xlabel('$\log(n)$', fontsize=14)
        plt.ylabel('$\log(R/S)$', fontsize=14)
        plt.title(f'R/S Аналіз: Показник Херста H = {H:.3f}', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('hurst_analysis.png', dpi=300)
        plt.show()
    
    return H, results
```

### Генерація тестових процесів

```python
def generate_fractional_brownian_motion(n: int, H: float, 
                                         method: str = 'davies_harte') -> np.ndarray:
    """
    Генерує фракційний броунівський рух з заданим показником Херста.
    
    Parameters:
    -----------
    n : int
        Кількість точок
    H : float
        Показник Херста (0 < H < 1)
    method : str
        Метод генерації ('davies_harte' або 'approximate')
    
    Returns:
    --------
    fbm : np.ndarray
        Траєкторія fBm
    """
    if method == 'approximate':
        # Апроксимація через суму гармонік
        t = np.linspace(0, 1, n)
        fbm = np.zeros(n)
        
        for k in range(1, 1000):  # Сума гармонік
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = (k ** (-H - 0.5)) * np.random.normal(0, 1)
            fbm += amplitude * np.sin(2 * np.pi * k * t + phase)
        
        # Нормалізація
        fbm = fbm / np.std(fbm) * np.sqrt(n)
        return np.cumsum(fbm)
    
    else:
        # Спрощена версія через накопичення корельованих приростів
        # Для точності потрібна більш складна реалізація
        increments = np.random.normal(0, 1, n)
        
        # Застосовуємо фільтр для створення кореляції
        # Це спрощена версія, для точності потрібен спектральний метод
        if H == 0.5:
            return np.cumsum(increments)
        else:
            # Апроксимація через AR процес
            phi = 2**(2*H - 1) - 1
            ar_process = np.zeros(n)
            for i in range(1, n):
                ar_process[i] = phi * ar_process[i-1] + increments[i]
            return ar_process

def generate_test_series(n: int = 1000) -> dict:
    """
    Генерує тестові часові ряди з різними значеннями H.
    """
    np.random.seed(42)
    
    series = {}
    
    # H = 0.3 (антиперсистентний)
    series['H_0.3'] = generate_fractional_brownian_motion(n, H=0.3)
    
    # H = 0.5 (броунівський рух)
    increments = np.random.normal(0, 1, n)
    series['H_0.5'] = np.cumsum(increments)
    
    # H = 0.7 (персистентний)
    series['H_0.7'] = generate_fractional_brownian_motion(n, H=0.7)
    
    # H = 0.9 (сильно персистентний)
    series['H_0.9'] = generate_fractional_brownian_motion(n, H=0.9)
    
    # AR(1) з φ=0.8 (має пам'ять, але не точно fBm)
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = 0.8 * ar1[i-1] + np.random.normal(0, 1)
    series['AR1_0.8'] = ar1
    
    return series

# Генерація тестових рядів
test_series = generate_test_series(n=2000)

# Візуалізація
fig, axes = plt.subplots(len(test_series), 1, figsize=(14, 12))

for idx, (name, series) in enumerate(test_series.items()):
    axes[idx].plot(series[:500], linewidth=1)  # Показуємо перші 500 точок
    axes[idx].set_ylabel(name, fontsize=10)
    axes[idx].grid(True, alpha=0.3)

axes[-1].set_xlabel('Час $t$', fontsize=12)
plt.suptitle('Тестові часові ряди з різними значеннями H', fontsize=14)
plt.tight_layout()
plt.savefig('test_series.png', dpi=300)
plt.show()
```

### Оцінка показника Херста для тестових рядів

```python
results_hurst = {}

for name, series in test_series.items():
    print(f"\n{'='*60}")
    print(f"Аналіз: {name}")
    print(f"{'='*60}")
    
    try:
        H, details = estimate_hurst(series, min_window=20, max_window=len(series)//4, 
                                     plot=False)
        results_hurst[name] = H
        
        print(f"Оцінка H: {H:.4f}")
        print(f"Фрактальна розмірність D = {2 - H:.4f}")
        print(f"R² = {details['r_squared']:.4f}")
        
        # Інтерпретація
        if abs(H - 0.5) < 0.1:
            interpretation = "Броунівський рух (немає пам'яті)"
        elif H < 0.5:
            interpretation = "Антиперсистентний (mean-reverting)"
        elif H < 0.7:
            interpretation = "Слабка персистентність"
        else:
            interpretation = "Сильна персистентність (є пам'ять)"
        
        print(f"Інтерпретація: {interpretation}")
        
    except Exception as e:
        print(f"Помилка: {e}")

# Порівняльна візуалізація
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, series) in enumerate(test_series.items()):
    if name in results_hurst:
        H_est = results_hurst[name]
        log_n, log_rs = rs_analysis(series, min_window=20, max_window=len(series)//4)
        
        axes[idx].scatter(log_n, log_rs, alpha=0.6, s=30)
        
        # Лінія регресії
        if len(log_n) > 1:
            slope, intercept, _, _, _ = stats.linregress(log_n, log_rs)
            log_n_fit = np.linspace(log_n.min(), log_n.max(), 100)
            log_rs_fit = slope * log_n_fit + intercept
            axes[idx].plot(log_n_fit, log_rs_fit, 'r-', linewidth=2)
        
        axes[idx].set_title(f'{name}\nH = {H_est:.3f}', fontsize=11)
        axes[idx].set_xlabel('$\log(n)$', fontsize=10)
        axes[idx].set_ylabel('$\log(R/S)$', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hurst_comparison.png', dpi=300)
plt.show()
```

### Практичний фільтр метрик за показником Херста

```python
class HurstFilter:
    """
    Фільтр метрик на основі показника Херста.
    Ігнорує метрики з H ≈ 0.5 (немає пам'яті) та фокусується на H > 0.7.
    """
    
    def __init__(self, min_hurst: float = 0.7, max_hurst: float = 1.0):
        """
        Parameters:
        -----------
        min_hurst : float
            Мінімальне значення H для включення метрики
        max_hurst : float
            Максимальне значення H для включення метрики
        """
        self.min_hurst = min_hurst
        self.max_hurst = max_hurst
        self.metrics_hurst = {}
    
    def evaluate_metric(self, metric_name: str, series: np.ndarray) -> dict:
        """
        Оцінює показник Херста для метрики.
        
        Returns:
        --------
        result : dict
            Словник з результатами оцінки
        """
        try:
            H, details = estimate_hurst(series, plot=False)
            
            result = {
                'metric': metric_name,
                'H': H,
                'fractal_dimension': 2 - H,
                'r_squared': details['r_squared'],
                'should_monitor': self.min_hurst <= H <= self.max_hurst,
                'interpretation': self._interpret_hurst(H)
            }
            
            self.metrics_hurst[metric_name] = result
            return result
            
        except Exception as e:
            return {
                'metric': metric_name,
                'error': str(e),
                'should_monitor': False
            }
    
    def _interpret_hurst(self, H: float) -> str:
        """Інтерпретує значення H."""
        if abs(H - 0.5) < 0.1:
            return "Броунівський рух (немає пам'яті, не передбачуваний)"
        elif H < 0.5:
            return "Антиперсистентний (mean-reverting)"
        elif H < 0.7:
            return "Слабка персистентність (короткострокова пам'ять)"
        else:
            return "Сильна персистентність (довгострокова пам'ять, передбачуваний)"
    
    def filter_metrics(self, metrics_dict: dict) -> dict:
        """
        Фільтрує словник метрик, залишаючи лише ті, що мають H > min_hurst.
        
        Parameters:
        -----------
        metrics_dict : dict
            Словник {metric_name: series}
        
        Returns:
        --------
        filtered : dict
            Відфільтровані метрики
        """
        filtered = {}
        
        for metric_name, series in metrics_dict.items():
            result = self.evaluate_metric(metric_name, series)
            
            if result.get('should_monitor', False):
                filtered[metric_name] = series
                print(f"✓ {metric_name}: H = {result['H']:.3f} - ВКЛЮЧЕНО")
            else:
                print(f"✗ {metric_name}: H = {result.get('H', 'N/A'):.3f} - ВИКЛЮЧЕНО")
        
        return filtered
    
    def get_report(self) -> pd.DataFrame:
        """Повертає звіт по всіх оцінених метриках."""
        if not self.metrics_hurst:
            return pd.DataFrame()
        
        data = []
        for metric, result in self.metrics_hurst.items():
            if 'error' not in result:
                data.append({
                    'Метрика': metric,
                    'H': result['H'],
                    'Фрактальна розмірність': result['fractal_dimension'],
                    'R²': result['r_squared'],
                    'Моніторити': 'Так' if result['should_monitor'] else 'Ні',
                    'Інтерпретація': result['interpretation']
                })
        
        return pd.DataFrame(data)

# Демонстрація фільтра
print("\n" + "="*60)
print("Демонстрація фільтра метрик за показником Херста")
print("="*60)

# Симуляція набору метрик
np.random.seed(42)
metrics = {
    'CPU_usage': generate_fractional_brownian_motion(1000, H=0.75),  # Має пам'ять
    'Memory_usage': generate_fractional_brownian_motion(1000, H=0.65),  # Слабка пам'ять
    'Network_latency': np.cumsum(np.random.normal(0, 1, 1000)),  # H ≈ 0.5
    'Disk_IO': generate_fractional_brownian_motion(1000, H=0.85),  # Сильна пам'ять
    'Random_noise': np.random.normal(0, 1, 1000),  # Білий шум (не накопичується)
}

# Створення фільтра (H > 0.7)
hurst_filter = HurstFilter(min_hurst=0.7, max_hurst=1.0)

# Фільтрація
filtered_metrics = hurst_filter.filter_metrics(metrics)

print(f"\nРезультат: {len(filtered_metrics)} з {len(metrics)} метрик включено")

# Звіт
report = hurst_filter.get_report()
print("\n" + "="*60)
print("Звіт по метриках:")
print("="*60)
print(report.to_string(index=False))
```

### Візуалізація порівняння процесів

```python
def compare_processes():
    """Порівняння процесів з різними значеннями H."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Генеруємо процеси
    np.random.seed(42)
    processes = {
        'H = 0.5 (Броунівський)': np.cumsum(np.random.normal(0, 1, 500)),
        'H = 0.7 (Персистентний)': generate_fractional_brownian_motion(500, H=0.7),
        'H = 0.9 (Сильно персистентний)': generate_fractional_brownian_motion(500, H=0.9),
        'AR(1), φ=0.8': None
    }
    
    # AR(1)
    ar1 = np.zeros(500)
    for i in range(1, 500):
        ar1[i] = 0.8 * ar1[i-1] + np.random.normal(0, 1)
    processes['AR(1), φ=0.8'] = ar1
    
    axes_flat = axes.flatten()
    
    for idx, (name, series) in enumerate(processes.items()):
        axes_flat[idx].plot(series, linewidth=1)
        axes_flat[idx].set_title(name, fontsize=12)
        axes_flat[idx].set_xlabel('Час $t$', fontsize=10)
        axes_flat[idx].set_ylabel('$X_t$', fontsize=10)
        axes_flat[idx].grid(True, alpha=0.3)
        
        # Оцінка H
        try:
            H, _ = estimate_hurst(series, plot=False)
            axes_flat[idx].text(0.02, 0.98, f'Оцінка H = {H:.3f}', 
                               transform=axes_flat[idx].transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            pass
    
    plt.suptitle('Порівняння процесів з різною пам\'яттю', fontsize=14)
    plt.tight_layout()
    plt.savefig('process_comparison.png', dpi=300)
    plt.show()

compare_processes()
```

## Висновки та наступні кроки

Ключові висновки:

1. **Показник Херста** $H$ кількісно вимірює "пам'ять" часового ряду
2. **$H = 0.5$** означає відсутність пам'яті (броунівський рух)
3. **$H > 0.7$** вказує на сильну персистентність — систему можна передбачати
4. **R/S аналіз** дозволяє обчислити $H$ для реальних даних
5. **Фільтрація метрик** за $H$ допомагає фокусуватися на передбачуваних сигналах

**Для SRE практики:**
- Метрики з $H \approx 0.5$ — випадковий шум, не варто витрачати ресурси на прогнозування
- Метрики з $H > 0.7$ — мають пам'ять, можна використовувати LSTM та інші методи з довгою історією
- Фільтр за $H$ зменшує шум в системі моніторингу та підвищує точність алертів

**Практичне застосування:** У [фінальному проекті предиктивного моніторингу](08_final_project_alerting.md) показник Херста використовується для фільтрації метрик — лише метрики з $H > 0.7$ включаються в систему алертингу, що значно зменшує кількість хибних спрацювань.

У наступних лекціях ми розглянемо, як нейронні мережі (LSTM) можуть використовувати цю пам'ять для передбачення майбутніх значень.

---

## Пов'язані теми

- **[MVP Предиктивного Моніторингу](08_final_project_alerting.md)** — практичне застосування показника Херста для фільтрації метрик у системі предиктивного алертингу

---

## Додаткові матеріали

### Рекомендована література

1. Mandelbrot, B. B., & Wallis, J. R. (1969). "Robustness of the rescaled range R/S in the measurement of noncyclic long run statistical dependence." *Water Resources Research*, 5(5), 967-988.
2. Peters, E. E. (1994). *Fractal Market Analysis: Applying Chaos Theory to Investment and Economics*. Wiley.

### Вправи для самостійної роботи

1. **Завдання 1:** Реалізуйте **детрендований R/S аналіз** (Detrended Fluctuation Analysis, DFA) як альтернативу класичному R/S. Порівняйте результати на одних і тих же даних.

2. **Завдання 2:** Створіть функцію, яка генерує **точний фракційний броунівський рух** через спектральний метод (метод Девіса-Харта). Порівняйте оцінки $H$ для точно згенерованого fBm та апроксимацій.

3. **Завдання 3:** Застосуйте R/S аналіз до реальних метрик з вашої системи (Prometheus, Grafana, або публічні датасети). Створіть дашборд, який показує:
   - Значення $H$ для кожної метрики
   - Рекомендації щодо моніторингу (моніторити/не моніторити)
   - Фрактальну розмірність

