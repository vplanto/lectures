"""
СЕМІНАР 5: «Динамічний кордон» (Anomaly Detection Lab)

Мета: Порівняти ефективність статичних та динамічних порогів (thresholds)

Практика: Реалізація пайплайну: Sliding Window -> LSTM -> Reconstruction Error

Завдання: Навчити модель на нормальних даних і протестувати її на рядах із 
          «тихими» аномаліями (наприклад, коли значення CPU в нормі, але 
          динаміка не відповідає очікуваній)

Результат: Побудова графіка, де аномалії виявляються через правило 3σ на 
           основі статистики помилок, а не абсолютних значень
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class SimpleLSTM(nn.Module):
    """
    Проста LSTM модель для виявлення аномалій.
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 50, 
                 num_layers: int = 2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class AnomalyDetector:
    """
    Детектор аномалій на основі помилки реконструкції LSTM.
    """
    
    def __init__(self, window_size: int = 60, hidden_size: int = 50,
                 num_layers: int = 2, device: str = None):
        """
        Ініціалізація детектора.
        
        Parameters:
        -----------
        window_size : int
            Розмір вікна для sliding window
        hidden_size : int
            Розмір прихованого шару LSTM
        num_layers : int
            Кількість шарів LSTM
        device : str, optional
            Пристрій для обчислень
        """
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.error_stats = None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Створення послідовностей для навчання."""
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)
    
    def train(self, series: pd.Series, epochs: int = 50, 
              batch_size: int = 32, train_ratio: float = 0.8) -> list:
        """
        Навчання моделі на нормальних даних.
        """
        # Нормалізація
        data = series.values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data).flatten()
        
        # Створення послідовностей
        X, y = self._create_sequences(data_scaled)
        
        # Розділення на train/test
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Перетворення в тензори
        X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(self.device)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(self.device)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(self.device)
        
        # Побудова моделі
        self.model = SimpleLSTM(input_size=1, hidden_size=self.hidden_size,
                               num_layers=self.num_layers).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Навчання
        history = []
        self.model.train()
        
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Обчислення втрат на тестовому наборі
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = criterion(test_outputs, y_test).item()
                history.append(test_loss)
        
        # Обчислення статистики помилок на нормальних даних
        with torch.no_grad():
            self.model.eval()
            train_outputs = self.model(X_train)
            train_errors = torch.abs(train_outputs - y_train).cpu().numpy()
            
            self.error_stats = {
                'mean': np.mean(train_errors),
                'std': np.std(train_errors),
                'threshold': np.mean(train_errors) + 3 * np.std(train_errors)
            }
        
        self.is_trained = True
        return history
    
    def detect_anomalies(self, series: pd.Series) -> Tuple[np.ndarray, Dict]:
        """
        Виявлення аномалій через помилку реконструкції.
        
        Returns:
        --------
        Tuple[np.ndarray, Dict]
            (errors, stats) - помилки та статистика
        """
        if not self.is_trained:
            raise ValueError("Модель не навчена!")
        
        # Нормалізація
        data = series.values.reshape(-1, 1)
        data_scaled = self.scaler.transform(data).flatten()
        
        # Створення послідовностей
        X, _ = self._create_sequences(data_scaled)
        
        # Прогнозування
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for i in range(len(X_tensor)):
                pred = self.model(X_tensor[i:i+1])
                predictions.append(pred.cpu().numpy()[0, 0])
        
        # Денормалізація
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        # Обчислення помилок
        actual = series.values[self.window_size:]
        errors = np.abs(actual - predictions)
        
        # Виявлення аномалій
        anomalies = errors > self.error_stats['threshold']
        
        stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'threshold': self.error_stats['threshold'],
            'n_anomalies': np.sum(anomalies),
            'anomaly_indices': np.where(anomalies)[0] + self.window_size
        }
        
        return errors, stats


def generate_data_with_silent_anomaly(n_points: int = 1000, 
                                     anomaly_start: int = 700,
                                     anomaly_duration: int = 50) -> pd.Series:
    """
    Генерація даних з 'тихою' аномалією.
    
    Тиха аномалія: значення в нормі, але динаміка не відповідає очікуваній.
    """
    np.random.seed(42)
    
    # Нормальний ряд (логістичне відображення)
    series = np.zeros(n_points)
    series[0] = 0.5
    r = 3.7
    
    for i in range(1, n_points):
        if anomaly_start <= i < anomaly_start + anomaly_duration:
            # Аномалія: зміна динаміки (інший параметр r)
            r_anomaly = 3.9
            series[i] = r_anomaly * series[i-1] * (1 - series[i-1])
        else:
            series[i] = r * series[i-1] * (1 - series[i-1])
        
        series[i] += np.random.normal(0, 0.01)
        series[i] = np.clip(series[i], 0, 1)
    
    return pd.Series(series, name='Series')


def compare_static_vs_dynamic_threshold(series: pd.Series, 
                                       anomaly_start: int = 700,
                                       anomaly_duration: int = 50):
    """
    Порівняння статичного та динамічного порогів.
    """
    print("="*80)
    print("ПОРІВНЯННЯ: СТАТИЧНИЙ vs ДИНАМІЧНИЙ ПОРІГ")
    print("="*80)
    
    # Статичний поріг (на основі абсолютних значень)
    static_threshold_high = np.percentile(series.values, 95)
    static_threshold_low = np.percentile(series.values, 5)
    
    static_anomalies = (series.values > static_threshold_high) | \
                       (series.values < static_threshold_low)
    
    print(f"\nСтатичний поріг:")
    print(f"  Верхній: {static_threshold_high:.4f} (95-й перцентиль)")
    print(f"  Нижній: {static_threshold_low:.4f} (5-й перцентиль)")
    print(f"  Виявлено аномалій: {np.sum(static_anomalies)}")
    
    # Динамічний поріг (на основі помилки реконструкції)
    print(f"\nДинамічний поріг (LSTM + Reconstruction Error):")
    
    # Навчання на нормальних даних (до аномалії)
    train_data = series[:anomaly_start]
    detector = AnomalyDetector(window_size=60, hidden_size=50, num_layers=2)
    
    print("  Навчання моделі на нормальних даних...")
    detector.train(train_data, epochs=50)
    
    print(f"  Статистика помилок на нормальних даних:")
    print(f"    Середнє: {detector.error_stats['mean']:.6f}")
    print(f"    Стандартне відхилення: {detector.error_stats['std']:.6f}")
    print(f"    Поріг (μ + 3σ): {detector.error_stats['threshold']:.6f}")
    
    # Виявлення аномалій на всьому ряді
    errors, stats = detector.detect_anomalies(series)
    
    print(f"  Виявлено аномалій: {stats['n_anomalies']}")
    
    # Візуалізація
    fig = plt.figure(figsize=(18, 12))
    
    # Графік 1: Оригінальний ряд зі статичним порогом
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(series.values, 'b-', linewidth=1, alpha=0.7, label='Часовий ряд')
    ax1.axhline(y=static_threshold_high, color='r', linestyle='--', 
               linewidth=2, label=f'Статичний поріг (95%)')
    ax1.axhline(y=static_threshold_low, color='r', linestyle='--', 
               linewidth=2, label=f'Статичний поріг (5%)')
    ax1.axvspan(anomaly_start, anomaly_start + anomaly_duration, 
               alpha=0.2, color='orange', label='Реальна аномалія')
    ax1.fill_between(range(len(series)), static_threshold_low, static_threshold_high,
                     where=static_anomalies, alpha=0.3, color='red', 
                     label='Виявлені аномалії (статичний)')
    ax1.set_xlabel('Час (t)')
    ax1.set_ylabel('Значення')
    ax1.set_title('Статичний поріг: Абсолютні значення', 
                 fontweight='bold', color='red')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Графік 2: Помилка реконструкції з динамічним порогом
    ax2 = plt.subplot(3, 2, 2)
    error_indices = np.arange(detector.window_size, len(series))
    ax2.plot(error_indices, errors, 'g-', linewidth=1.5, alpha=0.8, 
            label='Помилка реконструкції')
    ax2.axhline(y=stats['threshold'], color='r', linestyle='--', 
               linewidth=2, label=f'Динамічний поріг (μ + 3σ)')
    ax2.axvspan(anomaly_start, anomaly_start + anomaly_duration, 
               alpha=0.2, color='orange', label='Реальна аномалія')
    ax2.fill_between(error_indices, stats['threshold'], errors.max(),
                     where=(errors > stats['threshold']), 
                     alpha=0.3, color='red', 
                     label='Виявлені аномалії (динамічний)')
    ax2.set_xlabel('Час (t)')
    ax2.set_ylabel('Помилка реконструкції')
    ax2.set_title('Динамічний поріг: Помилка реконструкції', 
                 fontweight='bold', color='green')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Графік 3: Порівняння виявлення
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(series.values, 'b-', linewidth=1, alpha=0.5, label='Часовий ряд')
    ax3.axvspan(anomaly_start, anomaly_start + anomaly_duration, 
               alpha=0.3, color='orange', label='Реальна аномалія')
    
    # Статичні аномалії
    static_anomaly_indices = np.where(static_anomalies)[0]
    if len(static_anomaly_indices) > 0:
        ax3.scatter(static_anomaly_indices, series.values[static_anomaly_indices],
                   color='red', s=30, alpha=0.6, marker='x', 
                   label='Статичний поріг', zorder=5)
    
    # Динамічні аномалії
    if len(stats['anomaly_indices']) > 0:
        ax3.scatter(stats['anomaly_indices'], 
                   series.values[stats['anomaly_indices']],
                   color='green', s=30, alpha=0.6, marker='o', 
                   label='Динамічний поріг', zorder=5)
    
    ax3.set_xlabel('Час (t)')
    ax3.set_ylabel('Значення')
    ax3.set_title('Порівняння виявлення аномалій', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Графік 4: Розподіл помилок
    ax4 = plt.subplot(3, 2, 4)
    ax4.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(x=stats['threshold'], color='r', linestyle='--', 
               linewidth=2, label=f'Поріг (μ + 3σ)')
    ax4.axvline(x=stats['mean_error'], color='g', linestyle='--', 
               linewidth=2, label=f'Середнє (μ)')
    ax4.set_xlabel('Помилка реконструкції')
    ax4.set_ylabel('Частота')
    ax4.set_title('Розподіл помилок реконструкції', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Графік 5: Прогнозовані vs Реальні значення
    ax5 = plt.subplot(3, 2, 5)
    predictions = series.values[detector.window_size:] - errors * np.sign(
        series.values[detector.window_size:] - 
        (series.values[detector.window_size:] - errors)
    )
    # Спрощений підхід: показуємо реальні значення та помилки
    ax5.plot(error_indices, series.values[detector.window_size:], 
            'b-', linewidth=1, alpha=0.7, label='Реальні значення')
    ax5.plot(error_indices, series.values[detector.window_size:] - errors, 
            'g--', linewidth=1, alpha=0.7, label='Прогнозовані (приблизно)')
    ax5.axvspan(anomaly_start, anomaly_start + anomaly_duration, 
               alpha=0.2, color='orange')
    ax5.set_xlabel('Час (t)')
    ax5.set_ylabel('Значення')
    ax5.set_title('Реальні vs Прогнозовані значення', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Графік 6: Метрики порівняння
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    
    # Обчислення метрик
    true_anomaly_region = np.arange(anomaly_start, anomaly_start + anomaly_duration)
    
    # Точність виявлення для статичного порогу
    static_detected = np.intersect1d(static_anomaly_indices, true_anomaly_region)
    static_precision = len(static_detected) / len(static_anomaly_indices) if len(static_anomaly_indices) > 0 else 0
    static_recall = len(static_detected) / len(true_anomaly_region) if len(true_anomaly_region) > 0 else 0
    
    # Точність виявлення для динамічного порогу
    dynamic_detected = np.intersect1d(stats['anomaly_indices'], true_anomaly_region)
    dynamic_precision = len(dynamic_detected) / len(stats['anomaly_indices']) if len(stats['anomaly_indices']) > 0 else 0
    dynamic_recall = len(dynamic_detected) / len(true_anomaly_region) if len(true_anomaly_region) > 0 else 0
    
    text = f"МЕТРИКИ ПОРІВНЯННЯ\n\n"
    text += f"СТАТИЧНИЙ ПОРІГ:\n"
    text += f"  Precision: {static_precision:.2%}\n"
    text += f"  Recall: {static_recall:.2%}\n"
    text += f"  Виявлено: {len(static_anomaly_indices)} точок\n\n"
    text += f"ДИНАМІЧНИЙ ПОРІГ:\n"
    text += f"  Precision: {dynamic_precision:.2%}\n"
    text += f"  Recall: {dynamic_recall:.2%}\n"
    text += f"  Виявлено: {len(stats['anomaly_indices'])} точок\n\n"
    text += f"ВИСНОВОК:\n"
    if dynamic_recall > static_recall:
        text += f"  Динамічний поріг краще виявляє\n"
        text += f"  'тихі' аномалії (динаміка ≠ очікувана)"
    else:
        text += f"  Обидва методи мають обмеження"
    
    ax6.text(0.5, 0.5, text, ha='center', va='center', 
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            transform=ax6.transAxes)
    
    plt.tight_layout()
    plt.savefig('seminar_05_anomaly_detection.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено в файл: seminar_05_anomaly_detection.png")
    
    # Висновки
    print("\n" + "="*80)
    print("ВИСНОВКИ")
    print("="*80)
    print("\n1. Статичний поріг:")
    print("   - Працює тільки з абсолютними значеннями")
    print("   - Не виявляє 'тихі' аномалії (нормальне значення, але незвична динаміка)")
    
    print("\n2. Динамічний поріг (LSTM + Reconstruction Error):")
    print("   - Використовує помилку реконструкції, а не абсолютні значення")
    print("   - Виявляє аномалії через невідповідність очікуваній динаміці")
    print("   - Правило 3σ на основі статистики помилок")
    
    print("\n3. Переваги динамічного підходу:")
    print("   - Виявляє аномалії, які не видно в абсолютних значеннях")
    print("   - Адаптується до контексту та динаміки процесу")
    print("   - Ефективний для предиктивного моніторингу")
    
    plt.show()


def main():
    """
    Головна функція семінару.
    """
    print("="*80)
    print("СЕМІНАР 5: ДИНАМІЧНИЙ КОРДОН (ANOMALY DETECTION LAB)")
    print("="*80)
    
    # Генерація даних з 'тихою' аномалією
    print("\nГенерація даних з 'тихою' аномалією...")
    print("  - Нормальний ряд: логістичне відображення (r=3.7)")
    print("  - Аномалія (t=700-750): зміна динаміки (r=3.9)")
    print("  - Значення в нормі, але динаміка не відповідає очікуваній")
    
    series = generate_data_with_silent_anomaly(
        n_points=1000,
        anomaly_start=700,
        anomaly_duration=50
    )
    
    # Порівняння методів
    compare_static_vs_dynamic_threshold(series, anomaly_start=700, anomaly_duration=50)


if __name__ == "__main__":
    main()


