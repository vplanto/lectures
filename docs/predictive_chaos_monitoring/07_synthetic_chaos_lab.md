---
title: "07 Synthetic Chaos Lab"
type: lecture
module: Module 7
prerequisites: module 6
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# Генерація датасету через Логістичне відображення

## Парадокс: Чи можна передбачити хаос?

Детермінований хаос — це не випадковість. Кожен крок логістичного відображення $x_{n+1} = r x_n (1 - x_n)$ повністю визначений попереднім станом. Теоретично, якщо ми знаємо точне значення $x_0$ та параметр $r$, ми можемо обчислити $x_n$ для будь-якого $n$. 

Але на практиці:
- **Чутливість до початкових умов:** Малі помилки округлення експоненційно зростають
- **Стохастичний шум:** Реальні системи мають шум, який порушує детермінізм
- **Обмежена точність моделі:** LSTM не може точно відтворити хаотичну динаміку

Цей практикум демонструє межі передбачуваності: LSTM може навчитися короткостроковій динаміці, але ламається при додаванні стохастики або на довгих горизонтах.

## Математичний фундамент

### Логістичне відображення з шумом

**Детермінована версія:**
$$x_{n+1} = r \cdot x_n (1 - x_n)$$

**Стохастична версія:**
$$x_{n+1} = r \cdot x_n (1 - x_n) + \epsilon_n$$

Де $\epsilon_n \sim \mathcal{N}(0, \sigma^2)$ — білий шум.

**Вплив шуму:**
- При $\sigma \ll 1$: Хаотична динаміка зберігається, але з додатковими флуктуаціями
- При $\sigma \approx 0.1$: Шум починає домінувати над детермінованою структурою
- При $\sigma \gg 1$: Система стає переважно випадковою

### Передбачуваність хаосу

**Теоретична передбачуваність:**
Для детермінованого хаосу ($\sigma = 0$), якщо модель точно знає $f(x) = r x (1 - x)$, вона може передбачати на один крок точно.

**Практична передбачуваність:**
- **Короткострокова (1-5 кроків):** Можлива з високою точністю
- **Середньострокова (10-20 кроків):** Точність знижується через накопичення помилок
- **Довгострокова (>50 кроків):** Неможлива через експоненційне зростання помилок

**Експонента Ляпунова:**
$$\lambda = \lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^{n-1} \ln |f'(x_i)|$$

Якщо $\lambda > 0$, система хаотична, і помилки зростають як $e^{\lambda n}$.

### Обмеження LSTM для хаосу

**Проблеми:**
1. **Апроксимація нелінійності:** LSTM наближає $f(x)$ через складну нелінійну функцію, але не може точно відтворити її
2. **Накопичення помилок:** Кожна помилка на кроці $t$ впливає на всі наступні передбачення
3. **Шум:** Стохастичний шум робить передбачення принципово неможливим на довгих горизонтах

## Практична реалізація

### Генератор синтетичного хаосу

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ChaosGenerator:
    """
    Генератор часових рядів на основі логістичного відображення.
    """
    def __init__(self, r: float = 3.8, noise_level: float = 0.0, 
                 x0: float = None):
        """
        Parameters:
        -----------
        r : float
            Параметр контролю (0 <= r <= 4)
        noise_level : float
            Стандартне відхилення білого шуму
        x0 : float
            Початкова умова (None = випадкова)
        """
        self.r = r
        self.noise_level = noise_level
        self.x0 = x0 if x0 is not None else np.random.uniform(0.1, 0.9)
    
    def generate(self, n_samples: int, transient: int = 100) -> np.ndarray:
        """
        Генерує часовий ряд логістичного відображення.
        
        Parameters:
        -----------
        n_samples : int
            Кількість зразків для збереження
        transient : int
            Кількість ітерацій для "прогріву" (відкидаються)
        
        Returns:
        --------
        series : np.ndarray
            Часовий ряд
        """
        x = self.x0
        series = []
        
        # Пропускаємо transient
        for _ in range(transient):
            x = self.r * x * (1 - x)
            if self.noise_level > 0:
                x += np.random.normal(0, self.noise_level)
            x = np.clip(x, 0, 1)  # Забезпечуємо значення в [0, 1]
        
        # Генеруємо основну послідовність
        for _ in range(n_samples):
            x = self.r * x * (1 - x)
            if self.noise_level > 0:
                x += np.random.normal(0, self.noise_level)
            x = np.clip(x, 0, 1)
            series.append(x)
        
        return np.array(series)
    
    def generate_multiple(self, n_series: int, n_samples: int, 
                         transient: int = 100) -> np.ndarray:
        """
        Генерує кілька часових рядів з різними початковими умовами.
        """
        series_list = []
        for _ in range(n_series):
            self.x0 = np.random.uniform(0.1, 0.9)
            series = self.generate(n_samples, transient)
            series_list.append(series)
        return np.array(series_list)

# Демонстрація генерації
print("Генерація синтетичного хаосу...")
print("="*60)

# Різні рівні хаосу
chaos_levels = {
    'Стабільний (r=2.5)': 2.5,
    'Періодичний (r=3.2)': 3.2,
    'Хаотичний (r=3.8)': 3.8,
    'Сильно хаотичний (r=4.0)': 4.0
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, r) in enumerate(chaos_levels.items()):
    generator = ChaosGenerator(r=r, noise_level=0.0)
    series = generator.generate(n_samples=200, transient=100)
    
    axes[idx].plot(series, linewidth=1)
    axes[idx].set_title(f'{name}', fontsize=12)
    axes[idx].set_xlabel('Крок $n$', fontsize=10)
    axes[idx].set_ylabel('$x_n$', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Логістичне відображення для різних значень $r$', fontsize=14)
plt.tight_layout()
plt.savefig('chaos_levels.png', dpi=300)
plt.show()
```

### Вплив шуму на хаос

```python
def demonstrate_noise_impact():
    """
    Демонструє вплив шуму на хаотичну динаміку.
    """
    r = 3.8
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    
    fig, axes = plt.subplots(len(noise_levels), 1, figsize=(14, 12))
    
    for idx, noise in enumerate(noise_levels):
        generator = ChaosGenerator(r=r, noise_level=noise)
        series = generator.generate(n_samples=300, transient=100)
        
        axes[idx].plot(series, linewidth=1)
        axes[idx].set_ylabel(f'σ={noise}', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        
        if idx == 0:
            axes[idx].set_title('Вплив шуму на хаотичну динаміку (r=3.8)', fontsize=14)
    
    axes[-1].set_xlabel('Крок $n$', fontsize=12)
    plt.tight_layout()
    plt.savefig('noise_impact.png', dpi=300)
    plt.show()

demonstrate_noise_impact()
```

### Навчання LSTM на детермінованому хаосі

```python
class LSTMPredictor(nn.Module):
    """
    LSTM модель для передбачення наступного кроку логістичного відображення.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 50, 
                 num_layers: int = 2, output_size: int = 1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

def create_sequences(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Створює послідовності для навчання LSTM.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    
    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y).reshape(-1, 1)
    
    return X, y

def train_lstm_on_chaos(generator: ChaosGenerator, 
                        n_samples: int = 5000,
                        window_size: int = 20,
                        epochs: int = 100,
                        batch_size: int = 32) -> Tuple[nn.Module, List[float]]:
    """
    Навчає LSTM на синтетичному хаосі.
    """
    # Генерація даних
    print(f"Генерація даних (r={generator.r}, noise={generator.noise_level})...")
    data = generator.generate(n_samples=n_samples, transient=500)
    
    # Нормалізація
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Створення послідовностей
    X, y = create_sequences(data_scaled, window_size)
    
    # Розділення на train/val
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Конвертація в тензори
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Створення моделі
    model = LSTMPredictor(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Навчання
    train_losses = []
    val_losses = []
    
    print("Навчання моделі...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_epoch = 0
        indices = np.random.permutation(len(X_train_tensor))
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train_tensor[batch_indices]
            y_batch = y_train_tensor[batch_indices]
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_epoch += loss.item()
        
        train_loss_epoch /= (len(indices) // batch_size + 1)
        train_losses.append(train_loss_epoch)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor).item()
            val_losses.append(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_epoch:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
    
    # Зберігаємо scaler в моделі для подальшого використання
    model.scaler = scaler
    
    return model, train_losses, val_losses

# Навчання на детермінованому хаосі (без шуму)
print("\n" + "="*60)
print("Навчання LSTM на детермінованому хаосі (r=3.8, без шуму)")
print("="*60)

generator_det = ChaosGenerator(r=3.8, noise_level=0.0)
model_det, train_losses_det, val_losses_det = train_lstm_on_chaos(
    generator_det, n_samples=5000, window_size=20, epochs=100
)

# Візуалізація навчання
plt.figure(figsize=(12, 6))
plt.plot(train_losses_det, label='Train Loss', linewidth=2)
plt.plot(val_losses_det, label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Навчання LSTM на детермінованому хаосі', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_training_chaos.png', dpi=300)
plt.show()
```

### Тестування передбачуваності на різних горизонтах

```python
def test_prediction_horizon(model: nn.Module, generator: ChaosGenerator,
                           window_size: int = 20, max_horizon: int = 50,
                           n_test_samples: int = 1000) -> dict:
    """
    Тестує точність передбачення на різних часових горизонтах.
    
    Returns:
    --------
    results : dict
        Словник з метриками для кожного горизонту
    """
    # Генерація тестових даних
    test_data = generator.generate(n_samples=n_test_samples, transient=500)
    scaler = model.scaler
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    
    results = {
        'horizons': [],
        'mse': [],
        'mae': [],
        'rmse': []
    }
    
    model.eval()
    
    for horizon in range(1, max_horizon + 1):
        predictions = []
        actuals = []
        
        for i in range(window_size, len(test_data_scaled) - horizon):
            # Вхідне вікно
            X_input = test_data_scaled[i-window_size:i].reshape(1, window_size, 1)
            X_tensor = torch.FloatTensor(X_input)
            
            # Передбачення на horizon кроків вперед
            with torch.no_grad():
                # Однокрокове передбачення
                pred = model(X_tensor).item()
                
                # Багатокрокове передбачення (ітеративне)
                current_input = X_input[0, :, 0].copy()
                for h in range(horizon):
                    # Створюємо вікно з останніх window_size значень
                    window = current_input[-window_size:].reshape(1, window_size, 1)
                    window_tensor = torch.FloatTensor(window)
                    pred_step = model(window_tensor).item()
                    
                    # Оновлюємо вікно
                    current_input = np.append(current_input, pred_step)
                
                predictions.append(pred_step)
                actuals.append(test_data_scaled[i + horizon - 1])
        
        # Метрики
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        results['horizons'].append(horizon)
        results['mse'].append(mse)
        results['mae'].append(mae)
        results['rmse'].append(rmse)
    
    return results

# Тестування на детермінованому хаосі
print("\nТестування передбачуваності на різних горизонтах...")
results_det = test_prediction_horizon(model_det, generator_det, 
                                     window_size=20, max_horizon=50)

# Візуалізація
plt.figure(figsize=(12, 6))
plt.plot(results_det['horizons'], results_det['rmse'], 'o-', linewidth=2, markersize=4)
plt.xlabel('Горизонт передбачення (кроків)', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.title('Точність передбачення LSTM на детермінованому хаосі', fontsize=16)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('prediction_horizon_det.png', dpi=300)
plt.show()

print(f"\nRMSE на горизонті 1: {results_det['rmse'][0]:.6f}")
print(f"RMSE на горизонті 10: {results_det['rmse'][9]:.6f}")
print(f"RMSE на горизонті 50: {results_det['rmse'][49]:.6f}")
```

### Порівняння: детермінований хаос vs стохастичний

```python
# Навчання на стохастичному хаосі (з шумом)
print("\n" + "="*60)
print("Навчання LSTM на стохастичному хаосі (r=3.8, σ=0.05)")
print("="*60)

generator_stoch = ChaosGenerator(r=3.8, noise_level=0.05)
model_stoch, train_losses_stoch, val_losses_stoch = train_lstm_on_chaos(
    generator_stoch, n_samples=5000, window_size=20, epochs=100
)

# Тестування на стохастичному хаосі
results_stoch = test_prediction_horizon(model_stoch, generator_stoch,
                                       window_size=20, max_horizon=50)

# Порівняння
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses_det, label='Детермінований (train)', linewidth=2, alpha=0.7)
plt.plot(val_losses_det, label='Детермінований (val)', linewidth=2, linestyle='--', alpha=0.7)
plt.plot(train_losses_stoch, label='Стохастичний (train)', linewidth=2, alpha=0.7)
plt.plot(val_losses_stoch, label='Стохастичний (val)', linewidth=2, linestyle='--', alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Порівняння навчання', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(results_det['horizons'], results_det['rmse'], 'o-', 
        linewidth=2, markersize=4, label='Детермінований')
plt.plot(results_stoch['horizons'], results_stoch['rmse'], 's-',
        linewidth=2, markersize=4, label='Стохастичний (σ=0.05)')
plt.xlabel('Горизонт передбачення', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Точність передбачення', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig('deterministic_vs_stochastic.png', dpi=300)
plt.show()

print("\n" + "="*60)
print("Порівняння результатів:")
print("="*60)
print(f"Детермінований хаос:")
print(f"  RMSE (horizon=1): {results_det['rmse'][0]:.6f}")
print(f"  RMSE (horizon=50): {results_det['rmse'][49]:.6f}")
print(f"\nСтохастичний хаос (σ=0.05):")
print(f"  RMSE (horizon=1): {results_stoch['rmse'][0]:.6f}")
print(f"  RMSE (horizon=50): {results_stoch['rmse'][49]:.6f}")
print(f"\nВисновок: Шум значно погіршує передбачуваність на довгих горизонтах.")
```

### Візуалізація передбачень

```python
def visualize_predictions(model: nn.Module, generator: ChaosGenerator,
                         n_samples: int = 500, window_size: int = 20,
                         prediction_horizon: int = 1):
    """
    Візуалізує передбачення моделі на тестових даних.
    """
    # Генерація тестових даних
    test_data = generator.generate(n_samples=n_samples, transient=500)
    scaler = model.scaler
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
    
    # Передбачення
    predictions = []
    model.eval()
    
    for i in range(window_size, len(test_data_scaled) - prediction_horizon):
        X_input = test_data_scaled[i-window_size:i].reshape(1, window_size, 1)
        X_tensor = torch.FloatTensor(X_input)
        
        with torch.no_grad():
            pred = model(X_tensor).item()
            predictions.append(pred)
    
    # Денормалізація
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = test_data[window_size + prediction_horizon - 1:]
    
    # Візуалізація
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Повний ряд
    axes[0].plot(actuals[:300], 'b-', linewidth=1, label='Реальні значення', alpha=0.7)
    axes[0].plot(predictions[:300], 'r--', linewidth=2, label='Передбачення LSTM', alpha=0.8)
    axes[0].set_xlabel('Крок $n$', fontsize=12)
    axes[0].set_ylabel('$x_n$', fontsize=12)
    axes[0].set_title(f'Передбачення на {prediction_horizon} крок вперед', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Помилки
    errors = np.abs(actuals[:300] - predictions[:300])
    axes[1].plot(errors, 'g-', linewidth=1)
    axes[1].set_xlabel('Крок $n$', fontsize=12)
    axes[1].set_ylabel('Абсолютна помилка', fontsize=12)
    axes[1].set_title('Помилки передбачення', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Візуалізація для детермінованого хаосу
fig = visualize_predictions(model_det, generator_det, 
                          n_samples=500, window_size=20, prediction_horizon=1)
plt.savefig('predictions_deterministic.png', dpi=300)
plt.show()

# Візуалізація для стохастичного хаосу
fig = visualize_predictions(model_stoch, generator_stoch,
                           n_samples=500, window_size=20, prediction_horizon=1)
plt.savefig('predictions_stochastic.png', dpi=300)
plt.show()
```

### Аналіз меж передбачуваності

```python
def analyze_predictability_limits():
    """
    Аналізує межі передбачуваності для різних рівнів шуму.
    """
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    r = 3.8
    
    results = {}
    
    for noise in noise_levels:
        print(f"\nАналіз для σ = {noise}...")
        generator = ChaosGenerator(r=r, noise_level=noise)
        model, _, _ = train_lstm_on_chaos(
            generator, n_samples=3000, window_size=20, epochs=50
        )
        
        test_results = test_prediction_horizon(
            model, generator, window_size=20, max_horizon=30
        )
        
        results[noise] = test_results
    
    # Візуалізація
    plt.figure(figsize=(12, 6))
    for noise, test_results in results.items():
        plt.plot(test_results['horizons'], test_results['rmse'],
                'o-', linewidth=2, markersize=3, label=f'σ = {noise}')
    
    plt.xlabel('Горизонт передбачення', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.title('Вплив шуму на передбачуваність', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('predictability_limits.png', dpi=300)
    plt.show()
    
    return results

# Аналіз меж передбачуваності
print("\n" + "="*60)
print("Аналіз меж передбачуваності для різних рівнів шуму")
print("="*60)
predictability_results = analyze_predictability_limits()
```

### Robustness Testing: Data Poisoning та Concept Drift

**Проблема:** У реальних продакшн-системах моделі стикаються з двома критичними викликами:

1. **Data Poisoning (Отруєння даних):** Зловмисні або помилкові дані під час навчання можуть "отруїти" модель, змусивши її навчитися неправильним патернам
2. **Concept Drift (Зміна концепції):** Різка зміна поведінки системи (наприклад, після оновлення версії коду в Kubernetes) робить модель застарілою

**Для SRE:** Це критично важливо, бо:
- Kubernetes deployments часто змінюють поведінку системи
- Моніторинг має працювати навіть після оновлень
- Модель має виявляти, коли вона більше не відповідає реальності

```python
class RobustnessTester:
    """
    Клас для тестування стійкості моделі до data poisoning та concept drift.
    """
    def __init__(self, model: nn.Module, baseline_generator: ChaosGenerator):
        """
        Parameters:
        -----------
        model : nn.Module
            Навчена LSTM модель
        baseline_generator : ChaosGenerator
            Генератор для базової поведінки (нормальна робота)
        """
        self.model = model
        self.baseline_generator = baseline_generator
        self.baseline_performance = None
    
    def establish_baseline(self, n_samples: int = 1000, window_size: int = 20):
        """
        Встановлює базову продуктивність моделі на нормальних даних.
        """
        test_data = self.baseline_generator.generate(n_samples=n_samples, transient=500)
        scaler = self.model.scaler
        test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()
        
        predictions = []
        actuals = []
        self.model.eval()
        
        for i in range(window_size, len(test_data_scaled)):
            X_input = test_data_scaled[i-window_size:i].reshape(1, window_size, 1)
            X_tensor = torch.FloatTensor(X_input)
            
            with torch.no_grad():
                pred = self.model(X_tensor).item()
                predictions.append(pred)
                actuals.append(test_data_scaled[i])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        self.baseline_performance = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mean_error': np.mean(np.abs(actuals - predictions)),
            'std_error': np.std(np.abs(actuals - predictions))
        }
        
        print(f"\nБазова продуктивність встановлена:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Середня помилка: {self.baseline_performance['mean_error']:.6f}")
        print(f"  Стандартне відхилення помилки: {self.baseline_performance['std_error']:.6f}")
        
        return self.baseline_performance
    
    def test_data_poisoning(self, poisoning_ratio: float = 0.1, 
                           poisoning_type: str = 'adversarial',
                           n_samples: int = 1000, window_size: int = 20) -> dict:
        """
        Тестує стійкість до отруєння даних.
        
        Parameters:
        -----------
        poisoning_ratio : float
            Частка отруєних даних (0.0 - 1.0)
        poisoning_type : str
            Тип отруєння: 'adversarial', 'noise', 'outliers'
        n_samples : int
            Кількість тестових зразків
        window_size : int
            Розмір вікна
        
        Returns:
        --------
        results : dict
            Результати тестування
        """
        # Генерація нормальних даних
        normal_data = self.baseline_generator.generate(n_samples=n_samples, transient=500)
        scaler = self.model.scaler
        normal_data_scaled = scaler.transform(normal_data.reshape(-1, 1)).flatten()
        
        # Додавання отруєних даних
        n_poisoned = int(len(normal_data_scaled) * poisoning_ratio)
        poisoned_indices = np.random.choice(len(normal_data_scaled), n_poisoned, replace=False)
        
        poisoned_data = normal_data_scaled.copy()
        
        if poisoning_type == 'adversarial':
            # Адверсаріальне отруєння: дані, які виглядають нормально, але змінюють поведінку
            for idx in poisoned_indices:
                # Додаємо невеликий, але систематичний зсув
                poisoned_data[idx] += 0.2 * np.sin(2 * np.pi * idx / 50)
        
        elif poisoning_type == 'noise':
            # Високий шум
            for idx in poisoned_indices:
                poisoned_data[idx] += np.random.normal(0, 0.3)
        
        elif poisoning_type == 'outliers':
            # Викиди
            for idx in poisoned_indices:
                poisoned_data[idx] = np.random.choice([-1.0, 2.0])  # Екстремальні значення
        
        # Тестування
        predictions = []
        actuals = []
        self.model.eval()
        
        for i in range(window_size, len(poisoned_data)):
            X_input = poisoned_data[i-window_size:i].reshape(1, window_size, 1)
            X_tensor = torch.FloatTensor(X_input)
            
            with torch.no_grad():
                pred = self.model(X_tensor).item()
                predictions.append(pred)
                actuals.append(normal_data_scaled[i])  # Порівнюємо з нормальними даними
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # Порівняння з базовою продуктивністю
        if self.baseline_performance is not None:
            degradation = (rmse - self.baseline_performance['rmse']) / self.baseline_performance['rmse'] * 100
        else:
            degradation = 0.0
        
        results = {
            'poisoning_ratio': poisoning_ratio,
            'poisoning_type': poisoning_type,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'degradation_percent': degradation,
            'is_robust': degradation < 50.0  # Вважаємо стійким, якщо деградація < 50%
        }
        
        return results
    
    def test_concept_drift(self, drift_type: str = 'sudden', 
                          drift_magnitude: float = 0.5,
                          drift_start: int = 500,
                          n_samples: int = 1000, 
                          window_size: int = 20) -> dict:
        """
        Тестує стійкість до concept drift (зміни концепції).
        
        Parameters:
        -----------
        drift_type : str
            Тип drift: 'sudden' (раптова), 'gradual' (поступова), 'recurring' (повторювана)
        drift_magnitude : float
            Величина зміни (0.0 - 1.0)
        drift_start : int
            Позиція початку drift
        n_samples : int
            Кількість тестових зразків
        window_size : int
            Розмір вікна
        
        Returns:
        --------
        results : dict
            Результати тестування
        """
        # Генерація базових даних
        baseline_data = self.baseline_generator.generate(n_samples=n_samples, transient=500)
        scaler = self.model.scaler
        baseline_data_scaled = scaler.transform(baseline_data.reshape(-1, 1)).flatten()
        
        # Створення drift
        drifted_data = baseline_data_scaled.copy()
        
        if drift_type == 'sudden':
            # Раптова зміна (наприклад, оновлення версії коду)
            # Змінюємо параметр r в логістичному відображенні
            new_r = self.baseline_generator.r * (1 + drift_magnitude)
            drift_generator = ChaosGenerator(r=new_r, noise_level=self.baseline_generator.noise_level)
            drift_series = drift_generator.generate(n_samples=n_samples-drift_start, transient=0)
            drift_series_scaled = scaler.transform(drift_series.reshape(-1, 1)).flatten()
            drifted_data[drift_start:] = drift_series_scaled[:len(drifted_data[drift_start:])]
        
        elif drift_type == 'gradual':
            # Поступова зміна (наприклад, деградація системи)
            for i in range(drift_start, len(drifted_data)):
                progress = (i - drift_start) / (len(drifted_data) - drift_start)
                # Поступово змінюємо параметр
                current_r = self.baseline_generator.r * (1 + drift_magnitude * progress)
                temp_generator = ChaosGenerator(r=current_r, noise_level=self.baseline_generator.noise_level)
                # Генеруємо одне значення
                temp_series = temp_generator.generate(n_samples=1, transient=100)
                drifted_data[i] = scaler.transform(temp_series.reshape(-1, 1)).flatten()[0]
        
        elif drift_type == 'recurring':
            # Повторювана зміна (наприклад, періодичні оновлення)
            period = 200
            for cycle in range((len(drifted_data) - drift_start) // period):
                start_idx = drift_start + cycle * period
                end_idx = min(start_idx + period // 2, len(drifted_data))
                # Тимчасово змінюємо поведінку
                new_r = self.baseline_generator.r * (1 + drift_magnitude)
                drift_generator = ChaosGenerator(r=new_r, noise_level=self.baseline_generator.noise_level)
                drift_series = drift_generator.generate(n_samples=end_idx-start_idx, transient=0)
                drift_series_scaled = scaler.transform(drift_series.reshape(-1, 1)).flatten()
                drifted_data[start_idx:end_idx] = drift_series_scaled
        
        # Тестування
        predictions = []
        actuals = []
        errors_before_drift = []
        errors_after_drift = []
        
        self.model.eval()
        
        for i in range(window_size, len(drifted_data)):
            X_input = drifted_data[i-window_size:i].reshape(1, window_size, 1)
            X_tensor = torch.FloatTensor(X_input)
            
            with torch.no_grad():
                pred = self.model(X_tensor).item()
                predictions.append(pred)
                actuals.append(drifted_data[i])
                
                error = abs(pred - drifted_data[i])
                if i < drift_start:
                    errors_before_drift.append(error)
                else:
                    errors_after_drift.append(error)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Метрики до та після drift
        mse_before = mean_squared_error(actuals[:len(errors_before_drift)], predictions[:len(errors_before_drift)])
        mse_after = mean_squared_error(actuals[len(errors_before_drift):], predictions[len(errors_before_drift):])
        rmse_before = np.sqrt(mse_before)
        rmse_after = np.sqrt(mse_after)
        
        # Виявлення drift через зміну помилок
        mean_error_before = np.mean(errors_before_drift) if errors_before_drift else 0
        mean_error_after = np.mean(errors_after_drift) if errors_after_drift else 0
        error_increase = (mean_error_after - mean_error_before) / mean_error_before * 100 if mean_error_before > 0 else 0
        
        # Drift виявлено, якщо помилка зросла більш ніж на 50%
        drift_detected = error_increase > 50.0
        
        results = {
            'drift_type': drift_type,
            'drift_magnitude': drift_magnitude,
            'rmse_before': rmse_before,
            'rmse_after': rmse_after,
            'error_increase_percent': error_increase,
            'drift_detected': drift_detected,
            'mean_error_before': mean_error_before,
            'mean_error_after': mean_error_after
        }
        
        return results, drifted_data, predictions

# Демонстрація robustness testing
print("\n" + "="*60)
print("Robustness Testing: Data Poisoning та Concept Drift")
print("="*60)

# Використовуємо модель, навчену на детермінованому хаосі
tester = RobustnessTester(model_det, generator_det)

# Встановлюємо базову продуктивність
baseline = tester.establish_baseline(n_samples=1000, window_size=20)

# Тест 1: Data Poisoning
print("\n" + "="*60)
print("Тест 1: Data Poisoning (Отруєння даних)")
print("="*60)

poisoning_types = ['adversarial', 'noise', 'outliers']
poisoning_ratios = [0.05, 0.1, 0.2]

poisoning_results = []

for p_type in poisoning_types:
    for p_ratio in poisoning_ratios:
        result = tester.test_data_poisoning(
            poisoning_ratio=p_ratio,
            poisoning_type=p_type,
            n_samples=1000,
            window_size=20
        )
        poisoning_results.append(result)
        
        print(f"\n{p_type.capitalize()} poisoning ({p_ratio*100:.0f}%):")
        print(f"  RMSE: {result['rmse']:.6f}")
        print(f"  Деградація: {result['degradation_percent']:.2f}%")
        print(f"  Стійкість: {'✓' if result['is_robust'] else '✗'}")

# Візуалізація результатів data poisoning
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, p_type in enumerate(poisoning_types):
    type_results = [r for r in poisoning_results if r['poisoning_type'] == p_type]
    ratios = [r['poisoning_ratio'] * 100 for r in type_results]
    degradations = [r['degradation_percent'] for r in type_results]
    
    axes[idx].plot(ratios, degradations, 'o-', linewidth=2, markersize=8)
    axes[idx].axhline(y=50, color='r', linestyle='--', linewidth=1, label='Поріг (50%)')
    axes[idx].set_xlabel('Частка отруєних даних (%)', fontsize=11)
    axes[idx].set_ylabel('Деградація продуктивності (%)', fontsize=11)
    axes[idx].set_title(f'{p_type.capitalize()} Poisoning', fontsize=12)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_poisoning_test.png', dpi=300)
plt.show()

# Тест 2: Concept Drift
print("\n" + "="*60)
print("Тест 2: Concept Drift (Зміна концепції)")
print("="*60)

drift_types = ['sudden', 'gradual', 'recurring']
drift_magnitudes = [0.2, 0.5, 0.8]

drift_results = []

for d_type in drift_types:
    for d_mag in drift_magnitudes:
        result, drifted_data, predictions = tester.test_concept_drift(
            drift_type=d_type,
            drift_magnitude=d_mag,
            drift_start=500,
            n_samples=1000,
            window_size=20
        )
        drift_results.append((result, drifted_data, predictions))
        
        print(f"\n{d_type.capitalize()} drift (magnitude={d_mag}):")
        print(f"  RMSE до drift: {result['rmse_before']:.6f}")
        print(f"  RMSE після drift: {result['rmse_after']:.6f}")
        print(f"  Збільшення помилки: {result['error_increase_percent']:.2f}%")
        print(f"  Drift виявлено: {'✓' if result['drift_detected'] else '✗'}")

# Візуалізація concept drift
fig, axes = plt.subplots(len(drift_types), 1, figsize=(14, 12))

for idx, d_type in enumerate(drift_types):
    # Беремо результат з середньою величиною drift
    result, drifted_data, predictions = next(
        (r, d, p) for r, d, p in drift_results 
        if r['drift_type'] == d_type and abs(r['drift_magnitude'] - 0.5) < 0.1
    )
    
    # Відновлюємо оригінальні значення для візуалізації
    scaler = model_det.scaler
    drifted_data_orig = scaler.inverse_transform(drifted_data.reshape(-1, 1)).flatten()
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    axes[idx].plot(drifted_data_orig[:800], 'b-', linewidth=1, alpha=0.7, label='Реальні дані')
    axes[idx].plot(predictions_orig[:800], 'r--', linewidth=2, alpha=0.8, label='Передбачення')
    axes[idx].axvline(x=500, color='orange', linestyle='--', linewidth=2, label='Початок drift')
    axes[idx].set_xlabel('Час', fontsize=11)
    axes[idx].set_ylabel('Значення', fontsize=11)
    axes[idx].set_title(f'{d_type.capitalize()} Drift (magnitude=0.5, error increase={result["error_increase_percent"]:.1f}%)', fontsize=12)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('concept_drift_test.png', dpi=300)
plt.show()

# Порівняльна таблиця
print("\n" + "="*60)
print("Підсумок Robustness Testing")
print("="*60)

print("\nData Poisoning:")
print(f"{'Тип':<15} {'5%':<10} {'10%':<10} {'20%':<10}")
print("-" * 50)
for p_type in poisoning_types:
    type_results = [r for r in poisoning_results if r['poisoning_type'] == p_type]
    type_results.sort(key=lambda x: x['poisoning_ratio'])
    degradations = [f"{r['degradation_percent']:.1f}%" for r in type_results]
    print(f"{p_type.capitalize():<15} {degradations[0]:<10} {degradations[1]:<10} {degradations[2]:<10}")

print("\nConcept Drift:")
print(f"{'Тип':<15} {'Magnitude 0.2':<15} {'Magnitude 0.5':<15} {'Magnitude 0.8':<15}")
print("-" * 60)
for d_type in drift_types:
    type_results = [r for r, _, _ in drift_results if r['drift_type'] == d_type]
    type_results.sort(key=lambda x: x['drift_magnitude'])
    error_increases = [f"{r['error_increase_percent']:.1f}%" for r in type_results]
    print(f"{d_type.capitalize():<15} {error_increases[0]:<15} {error_increases[1]:<15} {error_increases[2]:<15}")

# Рекомендації
print("\n" + "="*60)
print("Рекомендації для SRE:")
print("="*60)
print("1. Моніторинг деградації продуктивності:")
print("   - Відстежуйте збільшення помилок передбачення")
print("   - Встановіть поріг (наприклад, +50%) для виявлення drift")
print("\n2. Адаптивне навчання:")
print("   - Періодично перенавчайте модель на нових даних")
print("   - Використовуйте online learning для поступових змін")
print("\n3. Захист від data poisoning:")
print("   - Валідуйте вхідні дані перед навчанням")
print("   - Використовуйте robust loss functions")
print("   - Видаляйте викиди та аномалії з навчального набору")
print("\n4. Виявлення concept drift:")
print("   - Відстежуйте статистику помилок у скільзному вікні")
print("   - Використовуйте статистичні тести (наприклад, KS-test)")
print("   - Автоматично тригерити перенавчання при виявленні drift")
```

## Висновки та наступні кроки

Ключові висновки:

1. **Детермінований хаос передбачуваний на коротких горизонтах:** LSTM може навчитися 1-5 кроків з високою точністю
2. **Довгострокова передбачуваність неможлива:** Помилки експоненційно зростають через чутливість до початкових умов
3. **Стохастичний шум руйнує передбачуваність:** Навіть невеликий шум ($\sigma = 0.05$) значно погіршує результати
4. **LSTM має межі:** Не може точно відтворити хаотичну динаміку на довгих горизонтах
5. **Robustness критична для продакшн:** Моделі мають бути стійкими до data poisoning та concept drift

**Для SRE практики:**
- Використовуйте короткострокові передбачення (1-10 кроків) для виявлення аномалій
- Не покладайтеся на довгострокові прогнози для хаотичних систем
- Комбінуйте детерміністичні моделі з стохастичними для реалістичних оцінок
- **Завжди тестуйте стійкість моделі до data poisoning та concept drift**
- **Відстежуйте деградацію продуктивності та автоматично виявляйте drift**
- **Періодично перенавчайте модель, особливо після оновлень Kubernetes**

У фінальному проекті ми застосуємо ці принципи до реальних даних Kubernetes.

---

## Додаткові матеріали

### Рекомендована література

1. Strogatz, S. H. (2014). *Nonlinear Dynamics and Chaos*. Westview Press.
2. Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis*. Cambridge University Press.

### Вправи для самостійної роботи

1. **Завдання 1:** Реалізуйте **експоненту Ляпунова** для логістичного відображення. Покажіть кореляцію між $\lambda$ та точністю передбачення LSTM.

2. **Завдання 2:** Дослідіть вплив **розміру вікна** на точність передбачення. Знайдіть оптимальне значення для різних рівнів хаосу ($r$).

3. **Завдання 3:** Створіть **ensemble метод**, який комбінує кілька LSTM моделей, навчених на різних початкових умовах. Порівняйте з одиночною моделлю.

4. **Завдання 4: Robustness Testing:** Реалізуйте систему автоматичного виявлення **concept drift** через моніторинг помилок передбачення у скільзному вікні. Створіть механізм автоматичного перенавчання моделі при виявленні drift. Застосуйте до симуляції оновлення версії коду в Kubernetes (раптова зміна параметра $r$ в логістичному відображенні).

5. **Завдання 5: Data Poisoning Defense:** Реалізуйте механізм захисту від **data poisoning**:
   - Виявлення та видалення викидів перед навчанням
   - Використання robust loss functions (наприклад, Huber loss)
   - Валідація вхідних даних через статистичні тести
   Порівняйте стійкість моделі з та без захисту.

