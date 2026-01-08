---
title: "MVP Предиктивного Моніторингу"
layout: default
nav_order: 8
parent: "Практикум (Workshop)"
---

# MVP Предиктивного Моніторингу

## Завдання: Знайти інцидент за 30 хвилин до падіння

Уявіть ситуацію: Kubernetes pod працює нормально, всі метрики в межах норми. Але через 30 хвилин він падає з помилкою OOM (Out of Memory). Класичний моніторинг виявить проблему лише після падіння. **Предиктивний моніторинг** має виявити аномалію **за 30 хвилин до** події, використовуючи тонкі зміни в динаміці метрик.

Цей фінальний проект об'єднує всі концепції курсу:
- **Хаос та нелінійність** в поведінці систем
- **Пам'ять процесів** (показник Херста)
- **LSTM** для передбачення
- **Помилка реконструкції** для виявлення аномалій
- **FFT** для видалення сезонності

## Структура проекту

### Етап 1: Підготовка даних

**Вхідні дані:**
- Анонімізований дамп метрик Kubernetes (CPU, Memory, Network, Disk I/O)
- Частота: 1 хвилина
- Період: 2 тижні
- Мітка: Час падіння pod (якщо є)

**Завдання:**
1. Завантажити та очистити дані
2. Визначити метрики з пам'яттю (показник Херста $H > 0.7$)
3. Видалити сезонність через FFT
4. Підготувати sliding windows для LSTM

### Етап 2: Навчання моделі

**Завдання:**
1. Навчити LSTM на нормальних даних (до інцидентів)
2. Обчислити статистику помилок реконструкції
3. Встановити динамічний поріг ($\mu + 3\sigma$)

### Етап 3: Виявлення аномалій

**Завдання:**
1. Застосувати модель до тестових даних
2. Виявити аномалії через помилку реконструкції
3. Оцінити, чи виявлено інцидент за 30+ хвилин до падіння

### Етап 4: Оцінка та впровадження

**Завдання:**
1. Обчислити метрики (Precision, Recall, F1)
2. Візуалізувати результати
3. Створити простий алертинг-пайплайн

## Реалізація повного пайплайну

### Клас для повного пайплайну

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Імпорт класів з попередніх лекцій
# (У реальному проекті вони будуть в окремих модулях)

class PredictiveMonitoringPipeline:
    """
    Повний пайплайн предиктивного моніторингу.
    """
    def __init__(self, window_size: int = 60, hidden_size: int = 50,
                 num_layers: int = 2, threshold_multiplier: float = 3.0):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.threshold_multiplier = threshold_multiplier
        
        self.models = {}  # Одна модель для кожної метрики
        self.scalers = {}
        self.error_stats = {}
        self.hurst_exponents = {}
        
    def compute_hurst(self, series: np.ndarray) -> float:
        """
        Обчислює показник Херста (спрощена версія).
        """
        from scipy import stats
        
        n = len(series)
        if n < 100:
            return 0.5  # За замовчуванням
        
        # R/S аналіз (спрощена версія)
        mean_series = np.mean(series)
        deviations = series - mean_series
        cumsum_deviations = np.cumsum(deviations)
        
        R = np.max(cumsum_deviations) - np.min(cumsum_deviations)
        S = np.std(series)
        
        if S < 1e-10:
            return 0.5
        
        # Логарифмічна регресія для оцінки H
        log_n = np.log(np.arange(10, min(100, n//2)))
        log_rs = []
        
        for window_size in np.arange(10, min(100, n//2), 5):
            n_windows = n // window_size
            if n_windows < 2:
                continue
            
            rs_values = []
            for i in range(n_windows):
                subseries = series[i*window_size:(i+1)*window_size]
                if len(subseries) < window_size:
                    continue
                
                sub_mean = np.mean(subseries)
                sub_deviations = subseries - sub_mean
                sub_cumsum = np.cumsum(sub_deviations)
                
                R_sub = np.max(sub_cumsum) - np.min(sub_cumsum)
                S_sub = np.std(subseries)
                
                if S_sub > 1e-10:
                    rs_values.append(R_sub / S_sub)
            
            if len(rs_values) > 0:
                log_rs.append(np.log(np.mean(rs_values)))
        
        if len(log_rs) < 2:
            return 0.5
        
        # Лінійна регресія
        slope, _, _, _, _ = stats.linregress(log_n[:len(log_rs)], log_rs)
        H = slope
        
        return max(0.0, min(1.0, H))  # Обмежуємо в [0, 1]
    
    def filter_metrics_by_hurst(self, metrics_df: pd.DataFrame, 
                                min_hurst: float = 0.7) -> List[str]:
        """
        Фільтрує метрики за показником Херста.
        """
        selected_metrics = []
        
        for column in metrics_df.columns:
            if column in ['timestamp', 'incident_time']:
                continue
            
            H = self.compute_hurst(metrics_df[column].values)
            self.hurst_exponents[column] = H
            
            if H >= min_hurst:
                selected_metrics.append(column)
                print(f"✓ {column}: H = {H:.3f} - ВКЛЮЧЕНО")
            else:
                print(f"✗ {column}: H = {H:.3f} - ВИКЛЮЧЕНО")
        
        return selected_metrics
    
    def remove_seasonality_fft(self, series: np.ndarray,
                              seasonal_periods: List[int] = [1440, 10080]) -> np.ndarray:
        """
        Видаляє сезонність через FFT (спрощена версія).
        """
        from scipy.fft import fft, ifft, fftfreq
        
        N = len(series)
        if N < max(seasonal_periods) * 2:
            return series  # Недостатньо даних для видалення сезонності
        
        # FFT
        fft_values = fft(series)
        frequencies = fftfreq(N)
        
        # Видаляємо сезонні частоти
        for period in seasonal_periods:
            target_freq = 1.0 / period
            # Знаходимо найближчу частоту
            idx = np.argmin(np.abs(frequencies - target_freq))
            if idx < len(fft_values):
                # Видаляємо основну частоту та гармоніки
                for harmonic in range(1, 5):
                    freq_idx = int(idx * harmonic)
                    if freq_idx < len(fft_values):
                        fft_values[freq_idx] = 0
                        fft_values[N - freq_idx] = 0  # Симетрія
        
        # Зворотне FFT
        deseasoned = np.real(ifft(fft_values))
        
        return deseasoned
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Створює послідовності для навчання LSTM.
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size])
        
        X = np.array(X).reshape(-1, self.window_size, 1)
        y = np.array(y).reshape(-1, 1)
        
        return X, y
    
    def fit(self, metrics_df: pd.DataFrame, 
            incident_column: Optional[str] = None,
            min_hurst: float = 0.7,
            remove_seasonality: bool = True,
            epochs: int = 100):
        """
        Навчає моделі на нормальних даних.
        """
        print("="*60)
        print("Етап 1: Фільтрація метрик за показником Херста")
        print("="*60)
        
        # Фільтрація метрик
        selected_metrics = self.filter_metrics_by_hurst(metrics_df, min_hurst)
        
        if len(selected_metrics) == 0:
            raise ValueError("Жодна метрика не пройшла фільтр за показником Херста!")
        
        print(f"\nВибрані метрики: {selected_metrics}")
        
        print("\n" + "="*60)
        print("Етап 2: Підготовка даних та навчання моделей")
        print("="*60)
        
        for metric in selected_metrics:
            print(f"\nОбробка метрики: {metric}")
            
            # Витягуємо дані
            data = metrics_df[metric].values
            
            # Видалення сезонності (опціонально)
            if remove_seasonality:
                data = self.remove_seasonality_fft(data)
                print(f"  Видалено сезонність")
            
            # Розділення на train/test (до інцидентів / після)
            if incident_column and incident_column in metrics_df.columns:
                incident_times = metrics_df[metrics_df[incident_column].notna()]
                if len(incident_times) > 0:
                    first_incident_idx = incident_times.index[0]
                    train_data = data[:first_incident_idx]
                else:
                    train_data = data[:int(0.8 * len(data))]
            else:
                train_data = data[:int(0.8 * len(data))]
            
            # Нормалізація
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
            self.scalers[metric] = scaler
            
            # Створення послідовностей
            X_train, y_train = self.create_sequences(train_scaled)
            
            if len(X_train) == 0:
                print(f"  Пропущено: недостатньо даних")
                continue
            
            # Конвертація в тензори
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            
            # Створення та навчання моделі
            model = nn.Sequential(
                nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True),
                nn.Linear(self.hidden_size, 1)
            )
            
            # Спрощена версія (для демонстрації)
            # У реальному проекті використовуйте LSTMPredictor з попередніх лекцій
            class SimpleLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(1, self.hidden_size, self.num_layers, 
                                       batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(self.hidden_size, 1)
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    return self.fc(lstm_out[:, -1, :])
            
            model = SimpleLSTM()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Навчання
            model.train()
            for epoch in range(epochs):
                indices = np.random.permutation(len(X_train_tensor))
                for i in range(0, len(indices), 32):
                    batch_indices = indices[i:i+32]
                    X_batch = X_train_tensor[batch_indices]
                    y_batch = y_train_tensor[batch_indices]
                    
                    optimizer.zero_grad()
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            
            self.models[metric] = model
            
            # Обчислення статистики помилок
            model.eval()
            with torch.no_grad():
                train_pred = model(X_train_tensor)
                errors = torch.abs(y_train_tensor - train_pred).numpy().flatten()
            
            self.error_stats[metric] = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'threshold': np.mean(errors) + self.threshold_multiplier * np.std(errors)
            }
            
            print(f"  Навчено. Поріг: {self.error_stats[metric]['threshold']:.6f}")
    
    def predict(self, metrics_df: pd.DataFrame,
               selected_metrics: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Передбачає значення та обчислює помилки реконструкції.
        """
        if selected_metrics is None:
            selected_metrics = list(self.models.keys())
        
        results = {}
        
        for metric in selected_metrics:
            if metric not in self.models:
                continue
            
            # Витягуємо дані
            data = metrics_df[metric].values
            
            # Видалення сезонності (якщо було застосовано при навчанні)
            # Тут можна додати логіку для видалення сезонності
            
            # Нормалізація
            scaler = self.scalers[metric]
            data_scaled = scaler.transform(data.reshape(-1, 1)).flatten()
            
            # Створення послідовностей
            X, y = self.create_sequences(data_scaled)
            
            if len(X) == 0:
                continue
            
            # Передбачення
            model = self.models[metric]
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions_scaled = model(X_tensor).numpy().flatten()
            
            # Денормалізація
            predictions = scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            
            # Обчислення помилок
            actual_values = data[self.window_size:]
            reconstruction_errors = np.abs(actual_values - predictions)
            
            results[metric] = {
                'predictions': predictions,
                'actual': actual_values,
                'errors': reconstruction_errors,
                'anomalies': (reconstruction_errors > self.error_stats[metric]['threshold']).astype(int)
            }
        
        return results
    
    def detect_incidents(self, metrics_df: pd.DataFrame,
                       incident_column: Optional[str] = None,
                       lead_time_minutes: int = 30) -> pd.DataFrame:
        """
        Виявляє інциденти за lead_time_minutes до фактичного падіння.
        """
        results = self.predict(metrics_df)
        
        # Об'єднуємо результати для всіх метрик
        all_anomalies = np.zeros(len(metrics_df))
        
        for metric, result in results.items():
            # Додаємо аномалії (з урахуванням window_size)
            anomaly_indices = np.where(result['anomalies'] == 1)[0] + self.window_size
            all_anomalies[anomaly_indices] = 1
        
        # Створюємо DataFrame з результатами
        output_df = metrics_df.copy()
        output_df['anomaly_detected'] = all_anomalies
        
        # Оцінка виявлення інцидентів
        if incident_column and incident_column in metrics_df.columns:
            incident_times = metrics_df[metrics_df[incident_column].notna()]
            
            detected_incidents = []
            for idx, row in incident_times.iterrows():
                incident_time = row[incident_column]
                incident_idx = metrics_df.index.get_loc(idx)
                
                # Перевіряємо, чи була виявлена аномалія за lead_time_minutes до інциденту
                detection_window_start = max(0, incident_idx - lead_time_minutes)
                detection_window_end = incident_idx
                
                if np.any(all_anomalies[detection_window_start:detection_window_end] == 1):
                    detected_incidents.append({
                        'incident_time': incident_time,
                        'detected': True,
                        'detection_time': metrics_df.index[
                            detection_window_start + np.argmax(
                                all_anomalies[detection_window_start:detection_window_end]
                            )
                        ]
                    })
                else:
                    detected_incidents.append({
                        'incident_time': incident_time,
                        'detected': False,
                        'detection_time': None
                    })
            
            print("\n" + "="*60)
            print("Результати виявлення інцидентів:")
            print("="*60)
            for incident in detected_incidents:
                status = "✓ ВИЯВЛЕНО" if incident['detected'] else "✗ НЕ ВИЯВЛЕНО"
                print(f"{status} - Інцидент: {incident['incident_time']}")
                if incident['detected']:
                    print(f"  Виявлено о: {incident['detection_time']}")
        
        return output_df
```

### Генерація синтетичних даних для демонстрації

```python
def generate_synthetic_kubernetes_metrics(n_samples: int = 10000,
                                        incident_times: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Генерує синтетичні метрики Kubernetes для демонстрації.
    """
    np.random.seed(42)
    
    # Базові метрики з сезонністю та трендом
    t = np.arange(n_samples)
    
    # CPU utilization
    cpu_base = 30 + 10 * np.sin(2 * np.pi * t / 1440)  # Денна сезонність
    cpu_trend = 0.001 * t
    cpu_noise = np.random.normal(0, 2, n_samples)
    cpu = cpu_base + cpu_trend + cpu_noise
    cpu = np.clip(cpu, 0, 100)
    
    # Memory usage
    memory_base = 50 + 15 * np.sin(2 * np.pi * t / 1440)
    memory_trend = 0.002 * t
    memory_noise = np.random.normal(0, 3, n_samples)
    memory = memory_base + memory_trend + memory_noise
    memory = np.clip(memory, 0, 100)
    
    # Network I/O
    network_base = 1000 + 200 * np.sin(2 * np.pi * t / 1440)
    network_trend = 0.1 * t
    network_noise = np.random.normal(0, 50, n_samples)
    network = network_base + network_trend + network_noise
    network = np.clip(network, 0, 10000)
    
    # Додаємо аномалії перед інцидентами
    if incident_times is None:
        incident_times = [5000, 8000]  # Приклади
    
    incident_labels = np.full(n_samples, np.nan)
    
    for incident_time in incident_times:
        # Позначаємо час інциденту
        incident_labels[incident_time] = incident_time
        
        # Додаємо аномалії за 30-60 хвилин до інциденту
        anomaly_start = max(0, incident_time - 60)
        anomaly_end = incident_time - 30
        
        # Зростання memory перед OOM
        memory[anomaly_start:anomaly_end] += np.linspace(0, 30, anomaly_end - anomaly_start)
        memory = np.clip(memory, 0, 100)
        
        # Зростання CPU через garbage collection
        cpu[anomaly_start:anomaly_end] += np.linspace(0, 20, anomaly_end - anomaly_start)
        cpu = np.clip(cpu, 0, 100)
    
    # Створюємо DataFrame
    timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='1min')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': cpu,
        'memory_usage': memory,
        'network_io': network,
        'incident_time': incident_labels
    })
    
    return df

# Генерація даних
print("Генерація синтетичних даних Kubernetes...")
metrics_df = generate_synthetic_kubernetes_metrics(
    n_samples=10000,
    incident_times=[5000, 8000]
)

print(f"Розмір датасету: {len(metrics_df)}")
print(f"Кількість інцидентів: {metrics_df['incident_time'].notna().sum()}")
```

### Запуск повного пайплайну

```python
# Створення пайплайну
pipeline = PredictiveMonitoringPipeline(
    window_size=60,
    hidden_size=50,
    num_layers=2,
    threshold_multiplier=3.0
)

# Навчання
pipeline.fit(
    metrics_df,
    incident_column='incident_time',
    min_hurst=0.6,  # Знижуємо для синтетичних даних
    remove_seasonality=True,
    epochs=50
)

# Виявлення інцидентів
results_df = pipeline.detect_incidents(
    metrics_df,
    incident_column='incident_time',
    lead_time_minutes=30
)

# Візуалізація
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# CPU
axes[0].plot(metrics_df['cpu_usage'], linewidth=1, alpha=0.7, label='CPU Usage')
axes[0].scatter(results_df[results_df['anomaly_detected'] == 1].index,
              results_df[results_df['anomaly_detected'] == 1]['cpu_usage'],
              color='red', s=20, zorder=5, label='Виявлені аномалії')
incident_indices = metrics_df[metrics_df['incident_time'].notna()].index
axes[0].axvline(x=incident_indices[0] if len(incident_indices) > 0 else 0,
               color='orange', linestyle='--', linewidth=2, label='Інцидент')
axes[0].set_ylabel('CPU Usage (%)', fontsize=12)
axes[0].set_title('Виявлення аномалій: CPU', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Memory
axes[1].plot(metrics_df['memory_usage'], linewidth=1, alpha=0.7, label='Memory Usage')
axes[1].scatter(results_df[results_df['anomaly_detected'] == 1].index,
               results_df[results_df['anomaly_detected'] == 1]['memory_usage'],
               color='red', s=20, zorder=5, label='Виявлені аномалії')
if len(incident_indices) > 0:
    axes[1].axvline(x=incident_indices[0], color='orange', linestyle='--', 
                   linewidth=2, label='Інцидент')
axes[1].set_ylabel('Memory Usage (%)', fontsize=12)
axes[1].set_title('Виявлення аномалій: Memory', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Аномалії
axes[2].fill_between(results_df.index, 0, results_df['anomaly_detected'],
                    alpha=0.5, color='red', label='Виявлені аномалії')
for idx in incident_indices:
    axes[2].axvline(x=idx, color='orange', linestyle='--', linewidth=2)
axes[2].set_xlabel('Час', fontsize=12)
axes[2].set_ylabel('Аномалія', fontsize=12)
axes[2].set_title('Загальні виявлені аномалії', fontsize=14)
axes[2].set_ylim(-0.1, 1.1)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('final_project_results.png', dpi=300)
plt.show()
```

## Висновки та наступні кроки

Ключові висновки:

1. **Повний пайплайн працює:** Комбінація фільтрації за Херстом, FFT, та LSTM дозволяє виявляти аномалії
2. **Раннє виявлення можливе:** Модель може виявити проблеми за 30+ хвилин до інциденту
3. **Важливість фільтрації:** Метрики з $H < 0.7$ не дають корисних сигналів
4. **Сезонність заважає:** Видалення сезонності покращує якість виявлення

**Для SRE практики:**
- Впроваджуйте поетапно: спочатку фільтрація метрик, потім LSTM, потім алертинг
- Калібруйте пороги на історичних даних
- Комбінуйте з класичними методами (thresholds) для кращого покриття

---

## Додаткові матеріали

### Рекомендована література

1. Hundman, K., et al. (2018). "Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding." *KDD*.
2. Siffer, A., et al. (2017). "Anomaly detection in streams with extreme value theory." *KDD*.

### Вправи для самостійної роботи

1. **Завдання 1:** Застосуйте пайплайн до реальних даних з Prometheus/Grafana. Порівняйте результати з класичними методами алертингу.

2. **Завдання 2:** Додайте **ensemble метод**, який комбінує результати від кількох метрик. Покажіть, що це покращує точність.

3. **Завдання 3:** Створіть **інтерактивний дашборд** (наприклад, через Streamlit або Plotly Dash) для візуалізації результатів у реальному часі.


