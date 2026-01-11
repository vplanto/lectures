---
title: "Від Forecasting до Anomaly Detection"
layout: default
nav_order: 5
parent: "Блок 3: Предиктивний Моніторинг (SRE Practice)"
---

# Від Forecasting до Anomaly Detection

## Парадокс: Чому нормальне значення може бути аномалією

Уявіть ситуацію: CPU utilization показує 45% — абсолютно нормальне значення для вашого сервера. Але LSTM, навчена на історичних даних, передбачала 30% з високою впевненістю. Різниця $|45\% - 30\%| = 15\%$ не виглядає критичною, але якщо це відбувається в контексті зростаючого тренду та невідповідає очікуваній динаміці — це **аномалія**.

Класичний підхід: "CPU > 80% = алерт" — не працює, бо ігнорує **контекст** та **динаміку**. Предиктивний моніторинг використовує не абсолютні значення, а **помилку реконструкції** — наскільки реальність відрізняється від очікуваного.

## Математичний фундамент

### Від Forecasting до Anomaly Detection

**Forecasting (Прогнозування):**
- Задача: Передбачити майбутнє значення $y_{t+1}$
- Модель: $f: (x_{t-k}, \ldots, x_t) \mapsto \hat{y}_{t+1}$
- Метрика: $\text{MSE} = \mathbb{E}[(y_{t+1} - \hat{y}_{t+1})^2]$

**Anomaly Detection (Виявлення аномалій):**
- Задача: Визначити, чи є $y_t$ аномалією
- Підхід: Навчити модель на нормальних даних, використати помилку реконструкції
- Метрика: $\text{Reconstruction Error} = |y_t - \hat{y}_t|$

### Помилка реконструкції (Reconstruction Error)

**Визначення:**

$$\text{RE}_t = |y_t - \hat{y}_t|$$

Де:
- $y_t$ — реальне значення на кроці $t$
- $\hat{y}_t$ — передбачене значення моделлю

**Нормалізована помилка реконструкції:**

$$\text{NRE}_t = \frac{|y_t - \hat{y}_t|}{\sigma_{\text{train}}}$$

Де $\sigma_{\text{train}}$ — стандартне відхилення помилок на навчальному наборі.

### Динамічний поріг на основі статистики помилок

**Статичний поріг (непрацюючий):**
$$\text{Alert if } |y_t| > \theta_{\text{static}}$$

**Динамічний поріг (працюючий):**
$$\text{Alert if } |y_t - \hat{y}_t| > \mu_{\text{error}} + k \cdot \sigma_{\text{error}}$$

Де:
- $\mu_{\text{error}}$ — середнє помилок на навчальному наборі
- $\sigma_{\text{error}}$ — стандартне відхилення помилок
- $k$ — множник (зазвичай $k = 3$ для $3\sigma$ правила)

**Інтерпретація $3\sigma$ правила:**
Якщо помилки розподілені нормально, то $P(|y_t - \hat{y}_t| > \mu + 3\sigma) \approx 0.0027$ (0.27% спостережень).

### Sliding Window для LSTM

**Вхідне вікно:**
$$X_t = [x_{t-w+1}, x_{t-w+2}, \ldots, x_t]$$

Де $w$ — розмір вікна (window size).

**Вихід:**
$$\hat{y}_{t+1} = \text{LSTM}(X_t)$$

**Параметри вікна:**
- $w$ повинно бути достатньо великим, щоб захопити залежності
- Типові значення: $w \in [20, 100]$ для метрик з частотою 1 хвилина
- Для логів за тиждень: $w \in [100, 1000]$

### Адаптивний поріг на основі рухомого середнього

**Проблема:** Статичний поріг $\mu + 3\sigma$ не враховує зміни в дисперсії помилок.

**Рішення:** Адаптивний поріг на основі останніх $N$ помилок:

$$\mu_t = \frac{1}{N} \sum_{i=t-N+1}^{t} |y_i - \hat{y}_i|$$

$$\sigma_t = \sqrt{\frac{1}{N-1} \sum_{i=t-N+1}^{t} (|y_i - \hat{y}_i| - \mu_t)^2}$$

$$\text{Threshold}_t = \mu_t + k \cdot \sigma_t$$

## Інженерна інтерпретація

### Пайплайн: Sliding Window Input -> LSTM -> Prediction

**Крок 1: Підготовка даних**
- Нормалізація (StandardScaler або MinMaxScaler)
- Створення sliding windows
- Розділення на train/validation/test

**Крок 2: Навчання LSTM**
- Навчання на нормальних даних (без аномалій)
- Модель навчається передбачати наступне значення
- Валідація на чистому тестовому наборі

**Крок 3: Обчислення помилок реконструкції**
- Для кожного спостереження: $\text{RE}_t = |y_t - \hat{y}_t|$
- Збір статистики: $\mu_{\text{error}}$, $\sigma_{\text{error}}$

**Крок 4: Встановлення динамічного порогу**
- $\text{Threshold} = \mu_{\text{error}} + 3\sigma_{\text{error}}$
- Або адаптивний поріг на основі останніх помилок

**Крок 5: Виявлення аномалій**
- Якщо $\text{RE}_t > \text{Threshold}$, то $y_t$ — аномалія

### Чому це працює краще за статичні пороги

**Приклад 1: Нормальне значення, але аномальна динаміка**
- CPU = 45% (нормально)
- Очікуване: 30% (тренд зниження)
- Помилка: 15% > $3\sigma$ → Аномалія виявлена

**Приклад 2: Високе значення, але очікуване**
- CPU = 85% (високо)
- Очікуване: 85% (сезонний сплеск)
- Помилка: 0% < $3\sigma$ → Не аномалія

**Висновок:** Аномалія визначається не абсолютним значенням, а **відхиленням від очікуваної динаміки**. Це принципово відрізняється від класичних методів, таких як лінійна регресія, яка не працює для нестаціонарних рядів через проблему спurious regression (детальніше в лекції про [випадкові блукання та гіпотезу ефективного ринку](01_random_walk_vs_memory.md)).

## Реалізація на Python

### Побудова пайплайну для Anomaly Detection

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class LSTMForecaster(nn.Module):
    """
    LSTM модель для прогнозування наступного значення часового ряду.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 50, 
                 num_layers: int = 2, output_size: int = 1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Використовуємо останній вихід
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class AnomalyDetector:
    """
    Система виявлення аномалій на основі LSTM та помилки реконструкції.
    """
    def __init__(self, window_size: int = 60, hidden_size: int = 50, 
                 num_layers: int = 2, threshold_multiplier: float = 3.0):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.threshold_multiplier = threshold_multiplier
        
        self.model = None
        self.scaler = StandardScaler()
        self.error_mean = None
        self.error_std = None
        self.threshold = None
    
    def create_sequences(self, data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створює послідовності для навчання LSTM.
        
        Parameters:
        -----------
        data : np.ndarray
            Часовий ряд (n_samples,)
        window_size : int
            Розмір вікна
        
        Returns:
        --------
        X : np.ndarray
            Вхідні послідовності (n_samples - window_size, window_size, 1)
        y : np.ndarray
            Цільові значення (n_samples - window_size, 1)
        """
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        
        X = np.array(X).reshape(-1, window_size, 1)
        y = np.array(y).reshape(-1, 1)
        
        return X, y
    
    def fit(self, train_data: np.ndarray, epochs: int = 100, 
            batch_size: int = 32, learning_rate: float = 0.001):
        """
        Навчає модель на нормальних даних.
        """
        # Нормалізація
        train_scaled = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
        
        # Створення послідовностей
        X_train, y_train = self.create_sequences(train_scaled, self.window_size)
        
        # Конвертація в тензори
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # Створення моделі
        self.model = LSTMForecaster(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1
        )
        
        # Навчання
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            # Batch training
            indices = np.random.permutation(len(X_train_tensor))
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Обчислення середньої помилки
            with torch.no_grad():
                train_pred = self.model(X_train_tensor)
                train_loss = criterion(train_pred, y_train_tensor).item()
                train_losses.append(train_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.6f}")
        
        # Обчислення помилок реконструкції на навчальному наборі
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_tensor)
            reconstruction_errors = torch.abs(y_train_tensor - train_pred).numpy().flatten()
        
        # Статистика помилок
        self.error_mean = np.mean(reconstruction_errors)
        self.error_std = np.std(reconstruction_errors)
        self.threshold = self.error_mean + self.threshold_multiplier * self.error_std
        
        print(f"\nСтатистика помилок реконструкції:")
        print(f"  Середнє: {self.error_mean:.6f}")
        print(f"  Стандартне відхилення: {self.error_std:.6f}")
        print(f"  Поріг ({self.threshold_multiplier}σ): {self.threshold:.6f}")
        
        return train_losses
    
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Передбачає значення та виявляє аномалії.
        
        Returns:
        --------
        predictions : np.ndarray
            Передбачені значення
        reconstruction_errors : np.ndarray
            Помилки реконструкції
        anomalies : np.ndarray
            Бінарний масив (1 = аномалія, 0 = норма)
        """
        # Нормалізація
        data_scaled = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Створення послідовностей
        X, y = self.create_sequences(data_scaled, self.window_size)
        
        # Передбачення
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions_scaled = self.model(X_tensor).numpy().flatten()
        
        # Денормалізація
        predictions = self.scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        # Обчислення помилок реконструкції
        actual_values = data[self.window_size:]
        reconstruction_errors = np.abs(actual_values - predictions)
        
        # Виявлення аномалій
        anomalies = (reconstruction_errors > self.threshold).astype(int)
        
        return predictions, reconstruction_errors, anomalies
    
    def predict_adaptive(self, data: np.ndarray, window_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Використовує адаптивний поріг на основі останніх помилок.
        """
        predictions, reconstruction_errors, _ = self.predict(data)
        
        # Адаптивний поріг
        adaptive_thresholds = np.zeros_like(reconstruction_errors)
        adaptive_anomalies = np.zeros_like(reconstruction_errors, dtype=int)
        
        for i in range(len(reconstruction_errors)):
            if i < window_size:
                # Використовуємо глобальний поріг для початку
                threshold = self.threshold
            else:
                # Обчислюємо локальну статистику
                recent_errors = reconstruction_errors[i-window_size:i]
                local_mean = np.mean(recent_errors)
                local_std = np.std(recent_errors)
                threshold = local_mean + self.threshold_multiplier * local_std
            
            adaptive_thresholds[i] = threshold
            adaptive_anomalies[i] = 1 if reconstruction_errors[i] > threshold else 0
        
        return predictions, reconstruction_errors, adaptive_anomalies
```

### Генерація синтетичного датасету з аномаліями

```python
def generate_synthetic_data_with_anomalies(n_samples: int = 2000, 
                                           anomaly_rate: float = 0.02,
                                           noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерує синтетичний часовий ряд з аномаліями.
    
    Returns:
    --------
    data : np.ndarray
        Часовий ряд з аномаліями
    labels : np.ndarray
        Бінарні мітки (1 = аномалія, 0 = норма)
    """
    # Базовий сигнал: синусоїда з трендом
    t = np.linspace(0, 4*np.pi, n_samples)
    base_signal = 50 + 10 * np.sin(t) + 0.01 * t  # Тренд + сезонність
    
    # Додаємо шум
    noise = np.random.normal(0, noise_level * np.std(base_signal), n_samples)
    data = base_signal + noise
    
    # Генеруємо аномалії
    labels = np.zeros(n_samples, dtype=int)
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Типи аномалій: сплеск, падіння, зміна тренду
        anomaly_type = np.random.choice(['spike', 'drop', 'trend_shift'])
        
        if anomaly_type == 'spike':
            data[idx] += np.random.uniform(20, 40)  # Позитивний сплеск
        elif anomaly_type == 'drop':
            data[idx] -= np.random.uniform(20, 40)  # Негативний сплеск
        else:
            # Зміна тренду на наступних 10 кроків
            end_idx = min(idx + 10, n_samples)
            trend = np.linspace(0, 15, end_idx - idx)
            data[idx:end_idx] += trend
        
        labels[idx] = 1
    
    return data, labels

# Генерація даних
np.random.seed(42)
data, true_labels = generate_synthetic_data_with_anomalies(
    n_samples=2000, 
    anomaly_rate=0.02,
    noise_level=0.1
)

# Розділення на train/test
train_size = int(0.7 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
test_labels = true_labels[train_size:]

print(f"Розмір навчального набору: {len(train_data)}")
print(f"Розмір тестового набору: {len(test_data)}")
print(f"Кількість аномалій у тесті: {np.sum(test_labels)}")
```

### Навчання та оцінка моделі

```python
# Створення детектора
detector = AnomalyDetector(
    window_size=60,
    hidden_size=50,
    num_layers=2,
    threshold_multiplier=3.0
)

# Навчання
print("Навчання моделі...")
train_losses = detector.fit(train_data, epochs=100, batch_size=32, learning_rate=0.001)

# Візуалізація навчання
plt.figure(figsize=(10, 6))
plt.plot(train_losses, linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.title('Навчання LSTM моделі', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_training_anomaly.png', dpi=300)
plt.show()

# Передбачення на тестовому наборі
predictions, errors, detected_anomalies = detector.predict(test_data)

# Оцінка
precision = precision_score(test_labels, detected_anomalies)
recall = recall_score(test_labels, detected_anomalies)
f1 = f1_score(test_labels, detected_anomalies)

print(f"\n{'='*60}")
print("Результати виявлення аномалій:")
print(f"{'='*60}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Поріг: {detector.threshold:.6f}")
```

### Статистична перевірка залишків: Тест Ljung-Box

**Проблема:** Як перевірити, чи наша LSTM модель витягла всю корисну інформацію з даних? Якщо в залишках (residuals) ще залишилася структура (автокореляція), це означає, що модель не використала всю доступну інформацію.

**Тест Ljung-Box** перевіряє гіпотезу про те, що залишки є незалежними (білий шум) проти альтернативи про наявність автокореляції.

**Математичний фундамент:**

Тест Ljung-Box базується на статистиці:

$$Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k}$$

Де:
- $n$ — кількість спостережень
- $h$ — максимальний лаг (зазвичай $\lfloor n/4 \rfloor$ або $\lfloor n/5 \rfloor$)
- $\hat{\rho}_k$ — вибіркова автокореляція залишків на лагу $k$

**Гіпотези:**
- $H_0$: Залишки не мають автокореляції (білий шум)
- $H_1$: Залишки мають автокореляцію (структура)

**Інтерпретація:**
- Якщо $p$-значення $> \alpha$ (наприклад, 0.05), то не відхиляємо $H_0$ → залишки є білим шумом → **модель добра**
- Якщо $p$-значення $\leq \alpha$, то відхиляємо $H_0$ → в залишках є структура → **модель можна покращити**

```python
def compute_residuals(model, X: np.ndarray, y: np.ndarray, scaler) -> np.ndarray:
    """
    Обчислює залишки (residuals) моделі.
    
    Parameters:
    -----------
    model : nn.Module
        Навчена LSTM модель
    X : np.ndarray
        Вхідні послідовності (n_samples, window_size, 1)
    y : np.ndarray
        Реальні значення (n_samples, 1)
    scaler : StandardScaler
        Scaler для денормалізації
    
    Returns:
    --------
    residuals : np.ndarray
        Залишки (actual - predicted)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions_scaled = model(X_tensor).numpy()
    
    # Денормалізація
    predictions = scaler.inverse_transform(predictions_scaled)
    actuals = scaler.inverse_transform(y)
    
    # Залишки
    residuals = (actuals - predictions).flatten()
    
    return residuals

def ljung_box_test(residuals: np.ndarray, lags: int = None, 
                   return_df: bool = False) -> dict:
    """
    Виконує тест Ljung-Box для перевірки автокореляції залишків.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Залишки моделі
    lags : int, optional
        Кількість лагів для тесту (за замовчуванням min(10, n/5))
    return_df : bool
        Чи повертати повну статистику по лагах
    
    Returns:
    --------
    result : dict
        Словник з результатами тесту
    """
    n = len(residuals)
    
    if lags is None:
        lags = min(10, n // 5)
    
    # Видаляємо NaN значення
    residuals_clean = residuals[~np.isnan(residuals)]
    
    if len(residuals_clean) < lags + 1:
        raise ValueError(f"Недостатньо даних для тесту. Потрібно мінімум {lags + 1} спостережень.")
    
    # Виконуємо тест Ljung-Box
    lb_result = acorr_ljungbox(residuals_clean, lags=lags, return_df=return_df)
    
    if return_df:
        # Повертаємо DataFrame з детальною статистикою
        return {
            'statistic': lb_result['lb_stat'].iloc[-1],
            'pvalue': lb_result['lb_pvalue'].iloc[-1],
            'lags': lags,
            'details': lb_result
        }
    else:
        # Повертаємо останнє значення (для всіх лагів разом)
        return {
            'statistic': lb_result[0][-1],
            'pvalue': lb_result[1][-1],
            'lags': lags
        }

def analyze_residuals(residuals: np.ndarray, lags: int = 20, 
                      alpha: float = 0.05) -> dict:
    """
    Комплексний аналіз залишків моделі.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Залишки моделі
    lags : int
        Кількість лагів для аналізу
    alpha : float
        Рівень значущості для тесту
    
    Returns:
    --------
    analysis : dict
        Словник з результатами аналізу
    """
    residuals_clean = residuals[~np.isnan(residuals)]
    
    # Тест Ljung-Box
    lb_result = ljung_box_test(residuals_clean, lags=min(lags, len(residuals_clean) // 5))
    
    # Обчислюємо автокореляцію
    autocorr = acf(residuals_clean, nlags=lags, fft=True)
    
    # Статистика залишків
    mean_residual = np.mean(residuals_clean)
    std_residual = np.std(residuals_clean)
    
    # Тест на нормальність (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(residuals_clean)
    
    # Інтерпретація
    is_white_noise = lb_result['pvalue'] > alpha
    interpretation = "Білий шум" if is_white_noise else "Наявна структура (автокореляція)"
    
    return {
        'ljung_box': lb_result,
        'autocorrelation': autocorr,
        'mean': mean_residual,
        'std': std_residual,
        'jarque_bera': {
            'statistic': jb_stat,
            'pvalue': jb_pvalue
        },
        'is_white_noise': is_white_noise,
        'interpretation': interpretation,
        'n_observations': len(residuals_clean)
    }

# Обчислення залишків на тестовому наборі
print("\n" + "="*60)
print("Статистична перевірка залишків моделі")
print("="*60)

# Створюємо послідовності для тестового набору
test_data_scaled = detector.scaler.transform(test_data.reshape(-1, 1)).flatten()
X_test, y_test_seq = detector.create_sequences(test_data_scaled, detector.window_size)

# Обчислюємо залишки
residuals = compute_residuals(detector.model, X_test, y_test_seq, detector.scaler)

# Аналіз залишків
residual_analysis = analyze_residuals(residuals, lags=20, alpha=0.05)

print(f"\nРезультати тесту Ljung-Box:")
print(f"  Статистика Q: {residual_analysis['ljung_box']['statistic']:.4f}")
print(f"  p-значення: {residual_analysis['ljung_box']['pvalue']:.6f}")
print(f"  Кількість лагів: {residual_analysis['ljung_box']['lags']}")
print(f"\nІнтерпретація: {residual_analysis['interpretation']}")

if residual_analysis['is_white_noise']:
    print("  ✓ Модель витягла всю корисну інформацію з даних")
    print("  ✓ Залишки не мають структури (білий шум)")
else:
    print("  ✗ В залишках залишилася структура")
    print("  ✗ Модель можна покращити (більше шарів, інша архітектура, додаткові ознаки)")

print(f"\nДодаткова статистика:")
print(f"  Середнє залишків: {residual_analysis['mean']:.6f} (має бути близько до 0)")
print(f"  Стандартне відхилення: {residual_analysis['std']:.6f}")
print(f"  Тест Jarque-Bera (нормальність): p = {residual_analysis['jarque_bera']['pvalue']:.6f}")
```

### Візуалізація аналізу залишків

```python
def visualize_residual_analysis(residuals: np.ndarray, analysis: dict, 
                                title: str = "Аналіз залишків моделі"):
    """
    Візуалізує результати аналізу залишків.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    residuals_clean = residuals[~np.isnan(residuals)]
    
    # Графік 1: Залишки в часі
    axes[0, 0].plot(residuals_clean[:500], 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('Час', fontsize=12)
    axes[0, 0].set_ylabel('Залишки', fontsize=12)
    axes[0, 0].set_title('Залишки в часі', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Додаємо інформацію про тест
    lb_pval = analysis['ljung_box']['pvalue']
    color = 'green' if lb_pval > 0.05 else 'red'
    axes[0, 0].text(0.02, 0.98, f"Ljung-Box: p = {lb_pval:.4f}", 
                   transform=axes[0, 0].transAxes,
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # Графік 2: Гістограма залишків
    axes[0, 1].hist(residuals_clean, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Додаємо нормальний розподіл для порівняння
    x_norm = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
    y_norm = stats.norm.pdf(x_norm, 
                           loc=analysis['mean'], 
                           scale=analysis['std'])
    axes[0, 1].plot(x_norm, y_norm, 'r-', linewidth=2, label='Нормальний розподіл')
    axes[0, 1].set_xlabel('Залишки', fontsize=12)
    axes[0, 1].set_ylabel('Щільність', fontsize=12)
    axes[0, 1].set_title('Розподіл залишків', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Графік 3: Автокореляційна функція (ACF)
    lags_plot = min(20, len(analysis['autocorrelation']) - 1)
    axes[1, 0].bar(range(1, lags_plot + 1), 
                   analysis['autocorrelation'][1:lags_plot + 1],
                   alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    # Довірчі інтервали (приблизно ±2/√n для білого шуму)
    conf_int = 1.96 / np.sqrt(len(residuals_clean))
    axes[1, 0].axhline(y=conf_int, color='r', linestyle='--', linewidth=1, alpha=0.7, label='95% CI')
    axes[1, 0].axhline(y=-conf_int, color='r', linestyle='--', linewidth=1, alpha=0.7)
    axes[1, 0].set_xlabel('Лаг $k$', fontsize=12)
    axes[1, 0].set_ylabel('Автокореляція $\\rho(k)$', fontsize=12)
    axes[1, 0].set_title('Автокореляційна функція залишків', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Графік 4: Q-Q plot (перевірка нормальності)
    stats.probplot(residuals_clean, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q plot (нормальність)', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    return fig

# Візуалізація
fig_residuals = visualize_residual_analysis(
    residuals, 
    residual_analysis,
    title=f"Аналіз залишків LSTM моделі (p = {residual_analysis['ljung_box']['pvalue']:.4f})"
)
plt.savefig('residual_analysis_lstm.png', dpi=300)
plt.show()
```

### Інтерпретація результатів та рекомендації

**Якщо тест Ljung-Box показує, що залишки є білим шумом ($p > 0.05$):**
- ✓ Модель успішно витягла всю корисну інформацію з даних
- ✓ Залишки не мають структури, яку можна було б використати для покращення прогнозу
- ✓ Модель готова до використання в продакшені

**Якщо тест показує наявність структури ($p \leq 0.05$):**
- ✗ В залишках залишилася автокореляція
- ✗ Модель не використала всю доступну інформацію
- **Рекомендації для покращення:**
  1. Збільшити кількість шарів LSTM або розмір прихованого стану
  2. Додати додаткові ознаки (сезонність, тренди, зовнішні фактори)
  3. Використати більше даних для навчання
  4. Спробувати інші архітектури (GRU, Transformer)
  5. Перевірити, чи правильно визначено вікно (window_size)

```python
# Приклад: порівняння двох моделей
print("\n" + "="*60)
print("Порівняння моделей за якістю залишків")
print("="*60)

# Модель 1: Поточна (hidden_size=50)
residuals_1 = residuals
analysis_1 = residual_analysis

# Модель 2: Більша модель (hidden_size=100)
detector_large = AnomalyDetector(
    window_size=60,
    hidden_size=100,  # Більша модель
    num_layers=2,
    threshold_multiplier=3.0
)

print("\nНавчання більшої моделі...")
detector_large.fit(train_data, epochs=100, batch_size=32, learning_rate=0.001)

# Обчислюємо залишки для більшої моделі
test_data_scaled_large = detector_large.scaler.transform(test_data.reshape(-1, 1)).flatten()
X_test_large, y_test_large = detector_large.create_sequences(
    test_data_scaled_large, detector_large.window_size
)
residuals_2 = compute_residuals(
    detector_large.model, X_test_large, y_test_large, detector_large.scaler
)
analysis_2 = analyze_residuals(residuals_2, lags=20, alpha=0.05)

# Порівняння
print(f"\nМодель 1 (hidden_size=50):")
print(f"  Ljung-Box p-value: {analysis_1['ljung_box']['pvalue']:.6f}")
print(f"  Інтерпретація: {analysis_1['interpretation']}")

print(f"\nМодель 2 (hidden_size=100):")
print(f"  Ljung-Box p-value: {analysis_2['ljung_box']['pvalue']:.6f}")
print(f"  Інтерпретація: {analysis_2['interpretation']}")

# Візуалізація порівняння
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF для моделі 1
lags_plot = min(20, len(analysis_1['autocorrelation']) - 1)
axes[0].bar(range(1, lags_plot + 1), 
            analysis_1['autocorrelation'][1:lags_plot + 1],
            alpha=0.7, color='blue', edgecolor='black')
conf_int = 1.96 / np.sqrt(len(residuals_1))
axes[0].axhline(y=conf_int, color='r', linestyle='--', linewidth=1, alpha=0.7)
axes[0].axhline(y=-conf_int, color='r', linestyle='--', linewidth=1, alpha=0.7)
axes[0].set_xlabel('Лаг $k$', fontsize=12)
axes[0].set_ylabel('Автокореляція', fontsize=12)
axes[0].set_title(f'Модель 1 (p = {analysis_1["ljung_box"]["pvalue"]:.4f})', fontsize=14)
axes[0].grid(True, alpha=0.3)

# ACF для моделі 2
lags_plot = min(20, len(analysis_2['autocorrelation']) - 1)
axes[1].bar(range(1, lags_plot + 1), 
            analysis_2['autocorrelation'][1:lags_plot + 1],
            alpha=0.7, color='green', edgecolor='black')
conf_int = 1.96 / np.sqrt(len(residuals_2))
axes[1].axhline(y=conf_int, color='r', linestyle='--', linewidth=1, alpha=0.7)
axes[1].axhline(y=-conf_int, color='r', linestyle='--', linewidth=1, alpha=0.7)
axes[1].set_xlabel('Лаг $k$', fontsize=12)
axes[1].set_ylabel('Автокореляція', fontsize=12)
axes[1].set_title(f'Модель 2 (p = {analysis_2["ljung_box"]["pvalue"]:.4f})', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison_residuals.png', dpi=300)
plt.show()
```

### Візуалізація результатів

```python
def visualize_anomaly_detection(data: np.ndarray, predictions: np.ndarray,
                                errors: np.ndarray, anomalies: np.ndarray,
                                true_labels: np.ndarray = None,
                                threshold: float = None,
                                title: str = "Виявлення аномалій"):
    """
    Візуалізує результати виявлення аномалій.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Графік 1: Дані та передбачення
    axes[0].plot(data, 'b-', linewidth=1, label='Реальні дані', alpha=0.7)
    axes[0].plot(predictions, 'r--', linewidth=2, label='Передбачення LSTM', alpha=0.8)
    
    # Позначаємо аномалії
    anomaly_indices = np.where(anomalies == 1)[0]
    if len(anomaly_indices) > 0:
        axes[0].scatter(anomaly_indices, data[anomaly_indices], 
                       color='red', s=50, zorder=5, label='Виявлені аномалії')
    
    # Позначаємо справжні аномалії (якщо є)
    if true_labels is not None:
        true_anomaly_indices = np.where(true_labels == 1)[0]
        if len(true_anomaly_indices) > 0:
            axes[0].scatter(true_anomaly_indices, data[true_anomaly_indices],
                           color='orange', s=30, marker='x', zorder=4, 
                           label='Справжні аномалії', alpha=0.7)
    
    axes[0].set_ylabel('Значення', fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Графік 2: Помилки реконструкції
    axes[1].plot(errors, 'g-', linewidth=1, label='Помилка реконструкції')
    if threshold is not None:
        axes[1].axhline(y=threshold, color='r', linestyle='--', 
                      linewidth=2, label=f'Поріг ({threshold:.4f})')
    axes[1].set_ylabel('Помилка', fontsize=12)
    axes[1].set_title('Помилка реконструкції', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Графік 3: Бінарні мітки
    axes[2].fill_between(range(len(anomalies)), 0, anomalies, 
                        alpha=0.5, color='red', label='Виявлені аномалії')
    if true_labels is not None:
        axes[2].fill_between(range(len(true_labels)), 0, true_labels,
                           alpha=0.3, color='orange', label='Справжні аномалії')
    axes[2].set_ylabel('Аномалія', fontsize=12)
    axes[2].set_xlabel('Час', fontsize=12)
    axes[2].set_title('Порівняння виявлених та справжніх аномалій', fontsize=14)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Візуалізація результатів
fig = visualize_anomaly_detection(
    test_data,
    predictions,
    errors,
    detected_anomalies,
    true_labels=test_labels,
    threshold=detector.threshold,
    title="Виявлення аномалій (статичний поріг)"
)
plt.savefig('anomaly_detection_static.png', dpi=300)
plt.show()
```

### Порівняння статичного та адаптивного порогів

```python
# Адаптивний поріг
predictions_adapt, errors_adapt, anomalies_adapt = detector.predict_adaptive(
    test_data, window_size=100
)

# Оцінка адаптивного підходу
precision_adapt = precision_score(test_labels, anomalies_adapt)
recall_adapt = recall_score(test_labels, anomalies_adapt)
f1_adapt = f1_score(test_labels, anomalies_adapt)

print(f"\n{'='*60}")
print("Порівняння статичного та адаптивного порогів:")
print(f"{'='*60}")
print(f"Статичний поріг:")
print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print(f"Адаптивний поріг:")
print(f"  Precision: {precision_adapt:.4f}, Recall: {recall_adapt:.4f}, F1: {f1_adapt:.4f}")

# Візуалізація адаптивного підходу
fig = visualize_anomaly_detection(
    test_data,
    predictions_adapt,
    errors_adapt,
    anomalies_adapt,
    true_labels=test_labels,
    threshold=None,  # Адаптивний поріг змінюється
    title="Виявлення аномалій (адаптивний поріг)"
)
plt.savefig('anomaly_detection_adaptive.png', dpi=300)
plt.show()
```

### Аналіз помилок: Precision-Recall крива

```python
from sklearn.metrics import precision_recall_curve, auc

def analyze_threshold_sensitivity(detector: AnomalyDetector, 
                                  test_data: np.ndarray,
                                  test_labels: np.ndarray):
    """
    Аналізує чутливість до вибору порогу.
    """
    # Отримуємо помилки реконструкції
    _, errors, _ = detector.predict(test_data)
    
    # Різні значення порогу
    thresholds = np.linspace(errors.min(), errors.max(), 100)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        anomalies = (errors > threshold).astype(int)
        
        if np.sum(anomalies) > 0:  # Уникаємо ділення на нуль
            precision = precision_score(test_labels, anomalies, zero_division=0)
            recall = recall_score(test_labels, anomalies, zero_division=0)
            f1 = f1_score(test_labels, anomalies, zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    # Precision-Recall крива
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, errors)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Візуалізація
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].plot(recalls, precisions, linewidth=2, label='Precision-Recall')
    axes[0].set_xlabel('Recall', fontsize=12)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title(f'Precision-Recall крива (AUC = {pr_auc:.4f})', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(thresholds, f1_scores, linewidth=2, label='F1-Score')
    axes[1].axvline(x=detector.threshold, color='r', linestyle='--', 
                   linewidth=2, label=f'Вибраний поріг ({detector.threshold:.4f})')
    axes[1].set_xlabel('Поріг', fontsize=12)
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('F1-Score в залежності від порогу', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis.png', dpi=300)
    plt.show()
    
    return thresholds, precisions, recalls, f1_scores

# Аналіз чутливості
analyze_threshold_sensitivity(detector, test_data, test_labels)
```

## Висновки та наступні кроки

Ключові висновки:

1. **Anomaly Detection відрізняється від Forecasting:** Використовуємо помилку реконструкції, а не абсолютні значення
2. **Динамічний поріг кращий за статичний:** $|y_t - \hat{y}_t| > \mu + 3\sigma$ враховує контекст
3. **Адаптивний поріг ще кращий:** Враховує зміни в дисперсії помилок з часом
4. **LSTM ефективна для виявлення аномалій:** Може захопити складні залежності в часових рядах
5. **Статистична перевірка залишків критична:** Тест Ljung-Box дозволяє перевірити, чи модель витягла всю корисну інформацію з даних

**Для SRE практики:**
- Використовуйте помилку реконструкції, а не абсолютні пороги
- Навчайте модель лише на нормальних даних
- Використовуйте адаптивні пороги для динамічних систем
- **Завжди перевіряйте залишки моделі тестом Ljung-Box** — якщо в залишках є структура, модель можна покращити
- Комбінуйте з іншими методами (FFT для сезонності, як у наступній лекції)

У наступній лекції ми розглянемо **FFT** для видалення сезонності та виявлення аномалій у частотній області.

---

## Пов'язані теми

- **[Випадкові блукання та Гіпотеза ефективного ринку](01_random_walk_vs_memory.md)** — пояснення, чому лінійна регресія не працює для нестаціонарних рядів та чому потрібні методи з урахуванням пам'яті процесу
- **[Робота в частотній області (Фур'є)](06_frequency_domain_fft.md)** — використання FFT для видалення сезонності перед навчанням LSTM моделей

---

## Додаткові матеріали

### Рекомендована література

1. Malhotra, P., et al. (2015). "Long short term memory networks for anomaly detection in time series." *ESANN*.
2. Hundman, K., et al. (2018). "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding." *KDD*.

### Вправи для самостійної роботи

1. **Завдання 1:** Реалізуйте **Autoencoder** для виявлення аномалій. Порівняйте з LSTM-based підходом на одних і тих же даних.

2. **Завдання 2:** Дослідіть вплив **розміру вікна** на якість виявлення аномалій. Знайдіть оптимальне значення для різних типів метрик.

3. **Завдання 3:** Створіть **ensemble метод**, який комбінує LSTM, статистичні методи (Z-score) та FFT-аналіз для покращення точності виявлення аномалій.

4. **Завдання 4:** Використайте **тест Ljung-Box** для порівняння різних архітектур LSTM (різна кількість шарів, розмір прихованого стану). Покажіть, як зміна архітектури впливає на якість залишків та чи можна покращити модель на основі результатів тесту.
