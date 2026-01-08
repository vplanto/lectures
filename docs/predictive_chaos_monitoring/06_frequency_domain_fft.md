---
title: "Робота в частотній області (Фур'є)"
layout: default
nav_order: 6
parent: "Блок 3: Предиктивний Моніторинг (SRE Practice)"
---

# Робота в частотній області (Фур'є)

## Парадокс: Чому "шум" сервера має музичну структуру

Уявіть спектрограму звуку сервера під навантаженням. Замість випадкового шуму ви бачите **гармоніки** — регулярні піки на частотах, кратних базовій. Це не випадковість — це математика. Кожен процес (CPU цикл, мережевий запит, операція з диском) має свою "ноту" у спектрі. Аномалія — це дисонанс, порушення гармонії.

Перехід у **частотну область** через **Fast Fourier Transform (FFT)** дозволяє:
1. Виявити **сезонність** (денні/тижневі цикли)
2. Видалити її, щоб оголити **аномальний сигнал**
3. Аналізувати **спектральні характеристики** "шуму" системи

## Математичний фундамент

### Дискретне перетворення Фур'є (DFT)

**Визначення:**

Для часового ряду $x[n]$ довжини $N$:

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}, \quad k = 0, 1, \ldots, N-1$$

Де:
- $X[k]$ — коефіцієнт Фур'є на частоті $f_k = k/N$
- $j = \sqrt{-1}$ — уявна одиниця
- $e^{-j 2\pi k n / N} = \cos(2\pi k n / N) - j \sin(2\pi k n / N)$

**Зворотне перетворення (IDFT):**

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j 2\pi k n / N}$$

### Спектральна щільність потужності (PSD)

**Періодограма:**

$$P[k] = \frac{1}{N} |X[k]|^2 = \frac{1}{N} (X[k] \cdot X^*[k])$$

Де $X^*[k]$ — комплексно спряжене до $X[k]$.

**Інтерпретація:**
- $P[k]$ показує "потужність" сигналу на частоті $f_k$
- Високі піки в $P[k]$ вказують на сильні періодичні компоненти

### Fast Fourier Transform (FFT)

**Алгоритм Cooley-Tukey (1965):**

FFT — це ефективна реалізація DFT зі складністю $O(N \log N)$ замість $O(N^2)$.

**Основна ідея:** Рекурсивне розбиття на парні та непарні індекси:

$$X[k] = X_{\text{even}}[k] + e^{-j 2\pi k / N} X_{\text{odd}}[k]$$

### Проблема спектрального розтікання (Spectral Leakage)

**Проблема:** При обчисленні FFT ми неявно припускаємо, що сигнал є **періодичним** з періодом $N$. Якщо це не так (що завжди в реальних даних), виникає **спектральне розтікання** — енергія однієї частоти "розтікається" на сусідні частоти.

**Математично:**

DFT обчислює коефіцієнти для сигналу, який повторюється нескінченно:

$$x_{\text{periodic}}[n] = \sum_{k=-\infty}^{\infty} x[n + kN]$$

Якщо реальний сигнал не є періодичним з періодом $N$, то на краях виникають **розриви**, які створюють високочастотні артефакти в спектрі.

**Візуалізація проблеми:**

```
Реальний сигнал:     [___/‾‾‾\___]
Періодичне продовження: [___/‾‾‾\___/‾‾‾\___/‾‾‾\___]
                        ↑              ↑
                    Розрив!       Розрив!
```

Ці розриви створюють "бокові пелюстки" (side lobes) у спектрі, що заважає точно визначити доминантні частоти.

### Віконні функції (Window Functions)

**Рішення:** Застосувати **віконну функцію** $w[n]$, яка плавно зменшує сигнал до нуля на краях, усуваючи розриви.

**Загальна формула:**

$$x_{\text{windowed}}[n] = x[n] \cdot w[n]$$

Де $w[n]$ — віконна функція, яка задовольняє:
- $w[0] \approx 0$ (початок)
- $w[N-1] \approx 0$ (кінець)
- $w[n] \approx 1$ для $n$ близько до центру

**Вікно Ханна (Hann Window):**

Одне з найпоширеніших вікон для спектрального аналізу:

$$w_{\text{Hann}}[n] = \frac{1}{2}\left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right), \quad n = 0, 1, \ldots, N-1$$

**Властивості:**
- Плавний перехід від 0 до 1 і назад до 0
- Добре пригнічує бокові пелюстки
- Компроміс між роздільною здатністю та пригніченням розтікання

**Вікно Геммінга (Hamming Window):**

$$w_{\text{Hamming}}[n] = 0.54 - 0.46\cos\left(\frac{2\pi n}{N-1}\right)$$

**Вікно Блекмана (Blackman Window):**

$$w_{\text{Blackman}}[n] = 0.42 - 0.5\cos\left(\frac{2\pi n}{N-1}\right) + 0.08\cos\left(\frac{4\pi n}{N-1}\right)$$

**Порівняння вікон:**

| Вікно | Пригнічення бокових пелюсток | Роздільна здатність | Застосування |
|-------|------------------------------|---------------------|--------------|
| Прямокутне (без вікна) | Погане (-13 dB) | Висока | Не рекомендується |
| Ханна | Добре (-32 dB) | Середня | Загальне використання |
| Геммінга | Добре (-43 dB) | Середня | Часто використовується |
| Блекмана | Відмінне (-58 dB) | Низька | Коли потрібне сильне пригнічення |

**Коли використовувати віконні функції:**

1. **Короткі вікна метрик:** Коли аналізуєте вікна довжиною < 1000 точок
2. **Неперіодичні сигнали:** Коли сигнал не є точно періодичним
3. **Висока точність:** Коли потрібно точно визначити частоти
4. **Аналіз аномалій:** Коли важливо не пропустити слабкі сигнали через розтікання

### Видалення сезонності через FFT

**Алгоритм:**

1. **Обчислення FFT:** $X[k] = \text{FFT}(x[n])$
2. **Ідентифікація сезонних частот:** Знайти $k$ з високими $|X[k]|$
3. **Фільтрація:** Встановити $X[k] = 0$ для сезонних частот
4. **Зворотне FFT:** $x_{\text{deseasoned}}[n] = \text{IFFT}(X[k])$

**Математично:**

$$x_{\text{deseasoned}}[n] = x[n] - \sum_{k \in S} \frac{1}{N} X[k] e^{j 2\pi k n / N}$$

Де $S$ — множина індексів сезонних частот.

### Спектральний аналіз для виявлення аномалій

**Підхід:**

1. **Базовий спектр:** Обчислити $P_{\text{normal}}[k]$ на нормальних даних
2. **Поточний спектр:** Обчислити $P_{\text{current}}[k]$ на поточному вікні
3. **Відхилення:** $\Delta P[k] = |P_{\text{current}}[k] - P_{\text{normal}}[k]|$
4. **Аномалія:** Якщо $\sum_{k} \Delta P[k] > \theta$, то виявлена аномалія

## Інженерна інтерпретація

### Сезонність у IT-метриках

**Типові патерни:**

1. **Денна сезонність:** Пік навантаження о 10:00, мінімум о 3:00
   - Період: 24 години = 1440 хвилин (якщо частота = 1 хв)
   - Частота: $f = 1/1440 \approx 0.000694$ Гц

2. **Тижнева сезонність:** Високе навантаження в робочі дні, низьке у вихідні
   - Період: 7 днів = 10080 хвилин
   - Частота: $f = 1/10080 \approx 0.000099$ Гц

3. **Годинна сезонність:** Cron jobs, scheduled tasks
   - Період: 60 хвилин
   - Частота: $f = 1/60 \approx 0.0167$ Гц

### Чому видалення сезонності важливе

**Проблема:** Сезонність може "маскувати" аномалії:
- Сплеск CPU о 10:00 може бути нормальним (денний пік)
- Але той самий сплеск о 3:00 — аномалія

**Рішення:** Видалити сезонність, щоб оголити **аномальний сигнал**:
- Після видалення сезонності сплеск о 3:00 стане очевидним
- LSTM може краще навчитися на детрендованих даних

### Спектральний аналіз "шуму" сервера

**Ідея:** Нормальний "шум" системи має характерний спектр. Аномалія змінює цей спектр.

**Застосування:**
- Виявлення нових процесів (нові піки в спектрі)
- Деградація компонентів (зміна амплітуд існуючих піків)
- Атаки (незвичайні частоти)

## Реалізація на Python

### Базові операції з FFT

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

def apply_window(timeseries: np.ndarray, window_type: str = 'hann') -> np.ndarray:
    """
    Застосовує віконну функцію до часового ряду.
    
    Parameters:
    -----------
    timeseries : np.ndarray
        Часовий ряд
    window_type : str
        Тип вікна: 'hann', 'hamming', 'blackman', 'rectangular'
    
    Returns:
    --------
    windowed : np.ndarray
        Часовий ряд після застосування вікна
    window : np.ndarray
        Сама віконна функція (для візуалізації)
    """
    N = len(timeseries)
    n = np.arange(N)
    
    if window_type.lower() == 'hann':
        window = 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    elif window_type.lower() == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    elif window_type.lower() == 'blackman':
        window = (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 
                  0.08 * np.cos(4 * np.pi * n / (N - 1)))
    elif window_type.lower() == 'rectangular':
        window = np.ones(N)
    else:
        raise ValueError(f"Невідомий тип вікна: {window_type}")
    
    windowed = timeseries * window
    
    return windowed, window

def compute_fft(timeseries: np.ndarray, sampling_rate: float = 1.0,
                window_type: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обчислює FFT часового ряду з опціональним застосуванням вікна.
    
    Parameters:
    -----------
    timeseries : np.ndarray
        Часовий ряд
    sampling_rate : float
        Частота дискретизації (наприклад, 1/60 для хвилинних даних)
    window_type : str, optional
        Тип вікна: 'hann', 'hamming', 'blackman', None (без вікна)
    
    Returns:
    --------
    frequencies : np.ndarray
        Масив частот
    fft_values : np.ndarray
        Комплексні коефіцієнти Фур'є
    """
    N = len(timeseries)
    
    # Застосування вікна (якщо вказано)
    if window_type is not None:
        timeseries, _ = apply_window(timeseries, window_type)
    
    # FFT
    fft_values = fft(timeseries)
    
    # Частоти
    frequencies = fftfreq(N, d=1/sampling_rate)
    
    # Беремо лише позитивні частоти (симетрія)
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    fft_values = fft_values[positive_freq_idx]
    
    return frequencies, fft_values

def compute_power_spectral_density(timeseries: np.ndarray, 
                                   sampling_rate: float = 1.0,
                                   window_type: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обчислює спектральну щільність потужності (PSD).
    
    Parameters:
    -----------
    timeseries : np.ndarray
        Часовий ряд
    sampling_rate : float
        Частота дискретизації
    window_type : str, optional
        Тип вікна для зменшення спектрального розтікання
    
    Returns:
    --------
    frequencies : np.ndarray
        Частоти
    psd : np.ndarray
        Спектральна щільність потужності
    """
    frequencies, fft_values = compute_fft(timeseries, sampling_rate, window_type)
    
    # PSD = |FFT|^2 / N
    # Для віконних функцій потрібна корекція енергії
    N = len(timeseries)
    if window_type is not None:
        # Корекція енергії вікна
        _, window = apply_window(np.ones(N), window_type)
        window_energy = np.sum(window ** 2)
        psd = np.abs(fft_values) ** 2 / window_energy
    else:
        psd = np.abs(fft_values) ** 2 / N
    
    return frequencies, psd

def plot_spectrum(timeseries: np.ndarray, sampling_rate: float = 1.0,
                  title: str = "Спектральний аналіз", 
                  window_type: Optional[str] = None):
    """
    Візуалізує часовий ряд та його спектр.
    
    Parameters:
    -----------
    timeseries : np.ndarray
        Часовий ряд
    sampling_rate : float
        Частота дискретизації
    title : str
        Заголовок графіка
    window_type : str, optional
        Тип вікна для зменшення спектрального розтікання
    """
    frequencies, psd = compute_power_spectral_density(timeseries, sampling_rate, window_type)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Часовий ряд
    axes[0].plot(timeseries, linewidth=1)
    axes[0].set_xlabel('Час', fontsize=12)
    axes[0].set_ylabel('Амплітуда', fontsize=12)
    axes[0].set_title(f'{title} - Часовий ряд', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Спектр (логарифмічна шкала)
    axes[1].semilogy(frequencies[1:], psd[1:], linewidth=1)  # Пропускаємо DC компонент
    axes[1].set_xlabel('Частота (Гц)', fontsize=12)
    axes[1].set_ylabel('PSD (логарифмічна шкала)', fontsize=12)
    axes[1].set_title(f'{title} - Спектральна щільність потужності', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes
```

### Генерація синтетичного датасету з сезонністю

```python
def generate_seasonal_timeseries(n_samples: int = 2000,
                                 daily_period: int = 1440,
                                 weekly_period: int = 10080,
                                 noise_level: float = 0.1,
                                 sampling_rate: float = 1.0) -> np.ndarray:
    """
    Генерує часовий ряд з денною та тижневою сезонністю.
    
    Parameters:
    -----------
    n_samples : int
        Кількість зразків
    daily_period : int
        Період денної сезонності (в одиницях sampling_rate)
    weekly_period : int
        Період тижневої сезонності
    noise_level : float
        Рівень шуму
    sampling_rate : float
        Частота дискретизації
    
    Returns:
    --------
    timeseries : np.ndarray
        Часовий ряд з сезонністю
    """
    t = np.arange(n_samples)
    
    # Денна сезонність (синусоїда з періодом daily_period)
    daily_component = 10 * np.sin(2 * np.pi * t / daily_period)
    
    # Тижнева сезонність (синусоїда з періодом weekly_period)
    weekly_component = 5 * np.sin(2 * np.pi * t / weekly_period)
    
    # Тренд
    trend = 0.01 * t
    
    # Базовий рівень
    base = 50
    
    # Шум
    noise = np.random.normal(0, noise_level * 10, n_samples)
    
    # Комбінація
    timeseries = base + trend + daily_component + weekly_component + noise
    
    return timeseries

# Генерація даних
np.random.seed(42)
seasonal_data = generate_seasonal_timeseries(
    n_samples=10000,
    daily_period=1440,  # 24 години (якщо sampling_rate = 1 хв)
    weekly_period=10080,  # 7 днів
    noise_level=0.1
)

# Візуалізація
fig, axes = plot_spectrum(seasonal_data[:5000], sampling_rate=1.0, 
                         title="Часовий ряд з сезонністю")
plt.savefig('seasonal_timeseries_spectrum.png', dpi=300)
plt.show()
```

### Демонстрація спектрального розтікання та віконних функцій

```python
def demonstrate_spectral_leakage():
    """
    Демонструє проблему спектрального розтікання та як віконні функції її вирішують.
    """
    # Генеруємо чистий синусоїдальний сигнал
    N = 256
    sampling_rate = 1.0
    t = np.arange(N)
    
    # Сигнал з частотою, яка НЕ є точною гармонікою базової частоти FFT
    # Це створює спектральне розтікання
    exact_freq = 10.0 / N  # Точна гармоніка (для порівняння)
    leaky_freq = 10.3 / N  # Не точна гармоніка (викликає розтікання)
    
    signal_exact = np.sin(2 * np.pi * exact_freq * t)
    signal_leaky = np.sin(2 * np.pi * leaky_freq * t)
    
    # Обчислюємо спектри без вікна та з вікном Ханна
    freq_exact, psd_exact = compute_power_spectral_density(
        signal_exact, sampling_rate, window_type=None
    )
    freq_leaky_no_window, psd_leaky_no_window = compute_power_spectral_density(
        signal_leaky, sampling_rate, window_type=None
    )
    freq_leaky_hann, psd_leaky_hann = compute_power_spectral_density(
        signal_leaky, sampling_rate, window_type='hann'
    )
    
    # Візуалізація
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Ряд 1: Точна частота (без розтікання)
    axes[0, 0].plot(t, signal_exact, linewidth=1)
    axes[0, 0].set_xlabel('Час', fontsize=11)
    axes[0, 0].set_ylabel('Амплітуда', fontsize=11)
    axes[0, 0].set_title('Сигнал з точною частотою', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].semilogy(freq_exact[1:50], psd_exact[1:50], linewidth=1, marker='o', markersize=3)
    axes[0, 1].axvline(x=exact_freq, color='r', linestyle='--', linewidth=1, label=f'f = {exact_freq:.4f}')
    axes[0, 1].set_xlabel('Частота (Гц)', fontsize=11)
    axes[0, 1].set_ylabel('PSD', fontsize=11)
    axes[0, 1].set_title('Спектр (без розтікання)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Ряд 2: Не точна частота БЕЗ вікна (з розтіканням)
    axes[1, 0].plot(t, signal_leaky, linewidth=1, color='orange')
    axes[1, 0].set_xlabel('Час', fontsize=11)
    axes[1, 0].set_ylabel('Амплітуда', fontsize=11)
    axes[1, 0].set_title('Сигнал з неточною частотою (без вікна)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(freq_leaky_no_window[1:50], psd_leaky_no_window[1:50], 
                       linewidth=1, color='orange', marker='o', markersize=3)
    axes[1, 1].axvline(x=leaky_freq, color='r', linestyle='--', linewidth=1, 
                      label=f'f = {leaky_freq:.4f}')
    axes[1, 1].set_xlabel('Частота (Гц)', fontsize=11)
    axes[1, 1].set_ylabel('PSD', fontsize=11)
    axes[1, 1].set_title('Спектр з розтіканням (без вікна)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Ряд 3: Не точна частота З вікном Ханна (розтікання зменшене)
    signal_leaky_windowed, window = apply_window(signal_leaky, 'hann')
    axes[2, 0].plot(t, signal_leaky_windowed, linewidth=1, color='green')
    axes[2, 0].plot(t, window, 'r--', linewidth=1, alpha=0.5, label='Вікно Ханна')
    axes[2, 0].set_xlabel('Час', fontsize=11)
    axes[2, 0].set_ylabel('Амплітуда', fontsize=11)
    axes[2, 0].set_title('Сигнал з вікном Ханна', fontsize=12)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].semilogy(freq_leaky_hann[1:50], psd_leaky_hann[1:50], 
                       linewidth=1, color='green', marker='o', markersize=3)
    axes[2, 1].axvline(x=leaky_freq, color='r', linestyle='--', linewidth=1, 
                      label=f'f = {leaky_freq:.4f}')
    axes[2, 1].set_xlabel('Частота (Гц)', fontsize=11)
    axes[2, 1].set_ylabel('PSD', fontsize=11)
    axes[2, 1].set_title('Спектр з вікном Ханна (розтікання зменшене)', fontsize=12)
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spectral_leakage_demonstration.png', dpi=300)
    plt.show()
    
    # Кількісне порівняння
    print("\n" + "="*60)
    print("Порівняння спектрального розтікання")
    print("="*60)
    
    # Знаходимо пік у спектрі
    peak_idx_no_window = np.argmax(psd_leaky_no_window[1:50])
    peak_idx_hann = np.argmax(psd_leaky_hann[1:50])
    
    # Обчислюємо ширину піку (на рівні -3dB)
    peak_power_no_window = psd_leaky_no_window[peak_idx_no_window + 1]
    peak_power_hann = psd_leaky_hann[peak_idx_hann + 1]
    
    threshold_no_window = peak_power_no_window / 2
    threshold_hann = peak_power_hann / 2
    
    # Ширина піку
    width_no_window = np.sum(psd_leaky_no_window[1:50] > threshold_no_window)
    width_hann = np.sum(psd_leaky_hann[1:50] > threshold_hann)
    
    print(f"\nБез вікна:")
    print(f"  Ширина піку (на -3dB): {width_no_window} бінів")
    print(f"  Максимальна потужність: {peak_power_no_window:.2e}")
    
    print(f"\nЗ вікном Ханна:")
    print(f"  Ширина піку (на -3dB): {width_hann} бінів")
    print(f"  Максимальна потужність: {peak_power_hann:.2e}")
    print(f"  Покращення (звуження піку): {width_no_window / width_hann:.2f}x")

# Демонстрація
demonstrate_spectral_leakage()
```

### Порівняння різних віконних функцій

```python
def compare_windows():
    """
    Порівнює різні віконні функції на прикладі короткого вікна метрик.
    """
    # Генеруємо короткий сигнал (симуляція короткого вікна метрик)
    N = 128  # Коротке вікно
    sampling_rate = 1.0
    t = np.arange(N)
    
    # Сигнал з двома близькими частотами
    f1 = 8.5 / N
    f2 = 9.2 / N
    signal = (np.sin(2 * np.pi * f1 * t) + 
              0.5 * np.sin(2 * np.pi * f2 * t) + 
              0.1 * np.random.randn(N))
    
    # Обчислюємо спектри з різними вікнами
    windows = ['rectangular', 'hann', 'hamming', 'blackman']
    window_names = ['Прямокутне (без вікна)', 'Ханна', 'Геммінга', 'Блекмана']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (window, name) in enumerate(zip(windows, window_names)):
        # Застосовуємо вікно
        signal_windowed, window_func = apply_window(signal, window)
        
        # Обчислюємо спектр
        freq, psd = compute_power_spectral_density(
            signal, sampling_rate, window_type=window if window != 'rectangular' else None
        )
        
        # Візуалізація
        axes[idx].semilogy(freq[1:40], psd[1:40], linewidth=2, label='PSD')
        axes[idx].axvline(x=f1, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'f1 = {f1:.4f}')
        axes[idx].axvline(x=f2, color='orange', linestyle='--', linewidth=1, alpha=0.7, label=f'f2 = {f2:.4f}')
        axes[idx].set_xlabel('Частота (Гц)', fontsize=11)
        axes[idx].set_ylabel('PSD (логарифмічна шкала)', fontsize=11)
        axes[idx].set_title(f'Вікно: {name}', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('window_comparison.png', dpi=300)
    plt.show()
    
    # Кількісне порівняння
    print("\n" + "="*60)
    print("Порівняння віконних функцій")
    print("="*60)
    
    results = {}
    for window, name in zip(windows, window_names):
        freq, psd = compute_power_spectral_density(
            signal, sampling_rate, window_type=window if window != 'rectangular' else None
        )
        
        # Знаходимо піки
        peaks, properties = signal.find_peaks(psd[1:40], height=np.max(psd[1:40]) * 0.1)
        peak_freqs = freq[peaks + 1]
        
        # Обчислюємо рівень бокових пелюсток (side lobe level)
        # Це відношення потужності найбільшої бокової пелюстки до основної
        sorted_psd = np.sort(psd[1:40])[::-1]
        if len(sorted_psd) > 1:
            side_lobe_level = 10 * np.log10(sorted_psd[1] / sorted_psd[0])
        else:
            side_lobe_level = -np.inf
        
        results[name] = {
            'peaks_found': len(peaks),
            'peak_frequencies': peak_freqs,
            'side_lobe_level_db': side_lobe_level
        }
    
    print("\nРезультати:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Знайдено піків: {result['peaks_found']}")
        print(f"  Частоти піків: {result['peak_frequencies']}")
        print(f"  Рівень бокових пелюсток: {result['side_lobe_level_db']:.2f} dB")

# Порівняння вікон
compare_windows()
```

### Застосування віконних функцій до реальних метрик

```python
def apply_windows_to_metrics(timeseries: np.ndarray, window_size: int = 256):
    """
    Демонструє застосування віконних функцій до коротких вікон метрик.
    """
    # Беремо коротке вікно
    if len(timeseries) < window_size:
        window_size = len(timeseries)
    
    window_data = timeseries[:window_size]
    
    # Обчислюємо спектри з різними вікнами
    windows_to_test = [None, 'hann', 'hamming', 'blackman']
    window_labels = ['Без вікна', 'Ханна', 'Геммінга', 'Блекмана']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (window, label) in enumerate(zip(windows_to_test, window_labels)):
        freq, psd = compute_power_spectral_density(
            window_data, sampling_rate=1.0, window_type=window
        )
        
        axes[idx].semilogy(freq[1:min(100, len(freq)-1)], 
                          psd[1:min(100, len(psd)-1)], 
                          linewidth=1.5)
        axes[idx].set_xlabel('Частота (Гц)', fontsize=11)
        axes[idx].set_ylabel('PSD', fontsize=11)
        axes[idx].set_title(f'Спектр: {label}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metrics_window_comparison.png', dpi=300)
    plt.show()
    
    print("\n" + "="*60)
    print("Рекомендації для коротких вікон метрик:")
    print("="*60)
    print("1. Для вікон < 500 точок: обов'язково використовуйте віконні функції")
    print("2. Вікно Ханна - хороший компроміс для більшості випадків")
    print("3. Вікно Блекмана - коли потрібне максимальне пригнічення розтікання")
    print("4. Вікно Геммінга - коли потрібна трохи краща роздільна здатність")

# Застосування до сезонних даних
apply_windows_to_metrics(seasonal_data, window_size=512)
```

### Видалення сезонності через FFT

```python
class SeasonalRemover:
    """
    Клас для видалення сезонності через FFT.
    """
    def __init__(self, seasonal_periods: List[int], 
                 threshold_ratio: float = 0.1):
        """
        Parameters:
        -----------
        seasonal_periods : List[int]
            Список періодів сезонності (наприклад, [1440, 10080])
        threshold_ratio : float
            Відношення для визначення сезонних частот
        """
        self.seasonal_periods = seasonal_periods
        self.threshold_ratio = threshold_ratio
    
    def find_seasonal_frequencies(self, timeseries: np.ndarray,
                                  sampling_rate: float = 1.0,
                                  window_type: Optional[str] = 'hann') -> List[int]:
        """
        Знаходить індекси частот, що відповідають сезонності.
        
        Parameters:
        -----------
        timeseries : np.ndarray
            Часовий ряд
        sampling_rate : float
            Частота дискретизації
        window_type : str, optional
            Тип вікна для зменшення спектрального розтікання
        """
        N = len(timeseries)
        frequencies, psd = compute_power_spectral_density(timeseries, sampling_rate, window_type)
        
        # Знаходимо індекси для сезонних частот
        seasonal_indices = []
        
        for period in self.seasonal_periods:
            # Частота, що відповідає періоду
            target_freq = 1.0 / period
            
            # Найближча частота в спектрі
            idx = np.argmin(np.abs(frequencies - target_freq))
            
            # Додаємо індекс та гармоніки
            seasonal_indices.append(idx)
            for harmonic in range(2, 5):  # Перші 4 гармоніки
                harmonic_freq = harmonic * target_freq
                if harmonic_freq < frequencies[-1]:
                    harmonic_idx = np.argmin(np.abs(frequencies - harmonic_freq))
                    seasonal_indices.append(harmonic_idx)
        
        return sorted(set(seasonal_indices))
    
    def remove_seasonality(self, timeseries: np.ndarray,
                          sampling_rate: float = 1.0,
                          window_type: Optional[str] = 'hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Видаляє сезонність з часового ряду.
        
        Parameters:
        -----------
        timeseries : np.ndarray
            Часовий ряд
        sampling_rate : float
            Частота дискретизації
        window_type : str, optional
            Тип вікна для зменшення спектрального розтікання (рекомендується 'hann')
        
        Returns:
        --------
        deseasoned : np.ndarray
            Часовий ряд без сезонності
        seasonal_component : np.ndarray
            Видалена сезонна компонента
        """
        N = len(timeseries)
        
        # Застосовуємо вікно перед FFT
        if window_type is not None:
            timeseries_windowed, _ = apply_window(timeseries, window_type)
        else:
            timeseries_windowed = timeseries
        
        # FFT
        fft_values = fft(timeseries_windowed)
        
        # Знаходимо сезонні частоти
        frequencies = fftfreq(N, d=1/sampling_rate)
        seasonal_indices = self.find_seasonal_frequencies(timeseries, sampling_rate, window_type)
        
        # Створюємо маску для сезонних частот
        mask = np.ones(N, dtype=bool)
        for idx in seasonal_indices:
            # Видаляємо позитивну та від'ємну частоту (симетрія)
            if idx < N:
                mask[idx] = False
            if N - idx < N:
                mask[N - idx] = False
        
        # Видаляємо сезонні компоненти
        fft_deseasoned = fft_values.copy()
        fft_deseasoned[~mask] = 0
        
        # Зворотне FFT для детрендованого ряду
        deseasoned = np.real(ifft(fft_deseasoned))
        
        # Сезонна компонента
        seasonal_component = timeseries - deseasoned
        
        return deseasoned, seasonal_component

# Видалення сезонності
remover = SeasonalRemover(seasonal_periods=[1440, 10080], threshold_ratio=0.1)
deseasoned, seasonal = remover.remove_seasonality(seasonal_data, sampling_rate=1.0)

# Візуалізація
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Оригінальний ряд
axes[0].plot(seasonal_data[:2000], linewidth=1, label='Оригінал')
axes[0].set_ylabel('Значення', fontsize=12)
axes[0].set_title('Оригінальний часовий ряд з сезонністю', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Сезонна компонента
axes[1].plot(seasonal[:2000], linewidth=1, color='orange', label='Сезонна компонента')
axes[1].set_ylabel('Значення', fontsize=12)
axes[1].set_title('Витягнута сезонна компонента', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Детрендований ряд
axes[2].plot(deseasoned[:2000], linewidth=1, color='green', label='Без сезонності')
axes[2].set_xlabel('Час', fontsize=12)
axes[2].set_ylabel('Значення', fontsize=12)
axes[2].set_title('Часовий ряд після видалення сезонності', fontsize=14)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('seasonal_removal.png', dpi=300)
plt.show()

# Порівняння спектрів
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

freq_orig, psd_orig = compute_power_spectral_density(seasonal_data[:5000])
freq_des, psd_des = compute_power_spectral_density(deseasoned[:5000])

axes[0].semilogy(freq_orig[1:100], psd_orig[1:100], linewidth=1, label='Оригінал')
axes[0].set_ylabel('PSD', fontsize=12)
axes[0].set_title('Спектр оригінального ряду', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].semilogy(freq_des[1:100], psd_des[1:100], linewidth=1, color='green', label='Без сезонності')
axes[1].set_xlabel('Частота (Гц)', fontsize=12)
axes[1].set_ylabel('PSD', fontsize=12)
axes[1].set_title('Спектр після видалення сезонності', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectrum_comparison.png', dpi=300)
plt.show()
```

### Спектральний аналіз для виявлення аномалій

```python
class SpectralAnomalyDetector:
    """
    Виявлення аномалій через спектральний аналіз.
    """
    def __init__(self, window_size: int = 1000, threshold_multiplier: float = 3.0):
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.baseline_psd = None
        self.psd_mean = None
        self.psd_std = None
    
    def fit(self, normal_data: np.ndarray, sampling_rate: float = 1.0):
        """
        Навчає модель на нормальних даних.
        """
        # Обчислюємо базовий спектр
        _, self.baseline_psd = compute_power_spectral_density(
            normal_data, sampling_rate
        )
        
        # Статистика PSD
        self.psd_mean = np.mean(self.baseline_psd)
        self.psd_std = np.std(self.baseline_psd)
        
        print(f"Базовий спектр обчислено:")
        print(f"  Середнє PSD: {self.psd_mean:.6f}")
        print(f"  Стандартне відхилення: {self.psd_std:.6f}")
    
    def detect(self, timeseries: np.ndarray, sampling_rate: float = 1.0) -> Tuple[float, bool]:
        """
        Виявляє аномалії в часовому ряді.
        
        Returns:
        --------
        anomaly_score : float
            Оцінка аномальності
        is_anomaly : bool
            Чи є аномалією
        """
        # Обчислюємо поточний спектр
        _, current_psd = compute_power_spectral_density(timeseries, sampling_rate)
        
        # Обрізаємо до розміру базового спектру
        min_len = min(len(self.baseline_psd), len(current_psd))
        baseline = self.baseline_psd[:min_len]
        current = current_psd[:min_len]
        
        # Обчислюємо відхилення
        deviation = np.abs(current - baseline)
        anomaly_score = np.sum(deviation)
        
        # Поріг
        threshold = self.psd_mean + self.threshold_multiplier * self.psd_std
        
        is_anomaly = anomaly_score > threshold
        
        return anomaly_score, is_anomaly
    
    def sliding_window_detection(self, timeseries: np.ndarray,
                                sampling_rate: float = 1.0,
                                step_size: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Виявлення аномалій у скільзному вікні.
        
        Returns:
        --------
        anomaly_scores : np.ndarray
            Оцінки аномальності для кожного вікна
        anomalies : np.ndarray
            Бінарні мітки аномалій
        """
        n_windows = (len(timeseries) - self.window_size) // step_size + 1
        anomaly_scores = []
        anomalies = []
        
        for i in range(0, len(timeseries) - self.window_size + 1, step_size):
            window = timeseries[i:i+self.window_size]
            score, is_anomaly = self.detect(window, sampling_rate)
            anomaly_scores.append(score)
            anomalies.append(1 if is_anomaly else 0)
        
        return np.array(anomaly_scores), np.array(anomalies)

# Генерація тестових даних з аномалією
normal_data = generate_seasonal_timeseries(n_samples=5000, noise_level=0.1)

# Додаємо аномалію (новий періодичний процес)
anomalous_data = normal_data.copy()
anomaly_start = 2500
anomaly_end = 3000
# Додаємо нову частоту (не сезонну)
t_anomaly = np.arange(anomaly_end - anomaly_start)
anomalous_data[anomaly_start:anomaly_end] += 15 * np.sin(2 * np.pi * t_anomaly / 100)

# Навчання детектора
detector = SpectralAnomalyDetector(window_size=1000, threshold_multiplier=3.0)
detector.fit(normal_data[:2000], sampling_rate=1.0)

# Виявлення аномалій
scores, detected = detector.sliding_window_detection(
    anomalous_data, sampling_rate=1.0, step_size=100
)

# Візуалізація
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Часовий ряд
axes[0].plot(anomalous_data, linewidth=1, label='Дані з аномалією')
axes[0].axvspan(anomaly_start, anomaly_end, alpha=0.3, color='red', label='Справжня аномалія')
axes[0].set_ylabel('Значення', fontsize=12)
axes[0].set_title('Часовий ряд з аномалією', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Оцінки аномальності
window_positions = np.arange(len(scores)) * 100
axes[1].plot(window_positions, scores, linewidth=2, label='Оцінка аномальності')
axes[1].axhline(y=detector.psd_mean + detector.threshold_multiplier * detector.psd_std,
               color='r', linestyle='--', linewidth=2, label='Поріг')
axes[1].set_ylabel('Оцінка аномальності', fontsize=12)
axes[1].set_title('Спектральна оцінка аномальності', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Виявлені аномалії
axes[2].fill_between(window_positions, 0, detected, alpha=0.5, color='red', label='Виявлені аномалії')
axes[2].set_xlabel('Час', fontsize=12)
axes[2].set_ylabel('Аномалія', fontsize=12)
axes[2].set_title('Виявлені аномалії', fontsize=14)
axes[2].set_ylim(-0.1, 1.1)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectral_anomaly_detection.png', dpi=300)
plt.show()
```

### Комбінація FFT та LSTM

```python
def combined_fft_lstm_pipeline(timeseries: np.ndarray, 
                               seasonal_periods: List[int],
                               window_size: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Комбінований пайплайн: видалення сезонності через FFT + LSTM.
    """
    # Крок 1: Видалення сезонності
    remover = SeasonalRemover(seasonal_periods=seasonal_periods)
    deseasoned, _ = remover.remove_seasonality(timeseries)
    
    # Крок 2: LSTM на детрендованих даних
    # Примітка: Тут можна використати AnomalyDetector з попередньої лекції
    # (05_anomaly_detection_strategies.md)
    # Для демонстрації використовуємо спрощений підхід
    
    # Розділення на train/test
    train_size = int(0.7 * len(deseasoned))
    train_data = deseasoned[:train_size]
    test_data = deseasoned[train_size:]
    
    # Спрощений приклад: обчислюємо помилки через просте порівняння
    # У реальному застосуванні тут була б LSTM модель
    train_mean = np.mean(train_data)
    train_std = np.std(train_data)
    
    # Простий детектор аномалій (для демонстрації)
    errors = np.abs(test_data - train_mean)
    threshold = train_mean + 3 * train_std
    anomalies = (errors > threshold).astype(int)
    
    return errors, anomalies

# Демонстрація комбінованого підходу
# Примітка: Для повної демонстрації потрібно імпортувати AnomalyDetector
# з попередньої лекції або реалізувати його локально
combined_errors, combined_anomalies = combined_fft_lstm_pipeline(
    seasonal_data,
    seasonal_periods=[1440, 10080],
    window_size=60
)

print(f"\nКомбінований підхід (FFT + простий детектор):")
print(f"  Виявлено аномалій: {np.sum(combined_anomalies)}")
print(f"  Середня помилка: {np.mean(combined_errors):.4f}")
print(f"\nПримітка: Для повної інтеграції з LSTM використайте")
print(f"AnomalyDetector з лекції 05_anomaly_detection_strategies.md")
```

## Висновки та наступні кроки

Ключові висновки:

1. **FFT дозволяє працювати в частотній області:** Виявлення періодичних компонентів
2. **Видалення сезонності важливе:** Оголяє аномальний сигнал, покращує навчання LSTM
3. **Спектральний аналіз для виявлення аномалій:** Порівняння поточного спектру з базовим
4. **Комбінація методів ефективніша:** FFT для сезонності + LSTM для залежностей
5. **Віконні функції критичні для коротких вікон:** Зменшують спектральне розтікання, покращують точність визначення частот

**Для SRE практики:**
- Використовуйте FFT для ідентифікації та видалення сезонності
- **Завжди застосовуйте віконні функції (особливо вікно Ханна) для коротких вікон метрик (< 1000 точок)**
- Комбінуйте FFT з LSTM для кращого виявлення аномалій
- Спектральний аналіз допомагає виявити нові процеси та деградацію
- Вікно Ханна — хороший компроміс між пригніченням розтікання та роздільною здатністю

**Інтеграція з аномалійним виявленням:** FFT має передувати навчанню LSTM моделей, оскільки видалення сезонності оголяє аномальний сигнал та покращує якість виявлення. Детальніше про це в лекції про [стратегії виявлення аномалій](05_anomaly_detection_strategies.md), де показано, як комбінувати FFT-денойзинг з LSTM для предиктивного моніторингу.

У наступних лекціях ми розглянемо практичні застосування цих методів на реальних даних.

---

## Пов'язані теми

- **[Від Forecasting до Anomaly Detection](05_anomaly_detection_strategies.md)** — інтеграція FFT для видалення сезонності перед навчанням LSTM моделей та покращення якості виявлення аномалій

---

## Додаткові матеріали

### Рекомендована література

1. Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing*. Prentice Hall.
2. Cooley, J. W., & Tukey, J. W. (1965). "An algorithm for the machine calculation of complex Fourier series." *Mathematics of computation*, 19(90), 297-301.

### Вправи для самостійної роботи

1. **Завдання 1:** Реалізуйте **Wavelet Transform** як альтернативу FFT. Порівняйте ефективність видалення сезонності через FFT та Wavelets.

2. **Завдання 2:** Створіть **спектрограму** (time-frequency representation) для аналізу нестаціонарних сигналів. Застосуйте до метрик зі змінною сезонністю.

3. **Завдання 3:** Реалізуйте **автоматичне виявлення сезонних періодів** через знаходження піків у спектрі. Порівняйте з ручним заданням періодів.

4. **Завдання 4:** Дослідіть вплив **різних віконних функцій** (Ханна, Геммінга, Блекмана) на точність визначення частот у коротких вікнах метрик. Покажіть, як вибір вікна впливає на здатність розрізняти близькі частоти.

