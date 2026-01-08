# Реалізація семінарів: Python модулі

Цей документ описує Python реалізацію трьох семінарів з курсу "Інтерпретований ШІ (XAI) та Прескриптивний Аналіз".

## Структура проєкту

```
prescriptive_xai_optimization/
├── seminar1_invariant_discovery.py    # Семінар 1: Виявлення інваріантів
├── seminar2_shap_lime.py              # Семінар 2: SHAP та LIME діагностика
├── seminar3_performance_doctor.py    # Семінар 3: Прескриптивний аналіз
├── example_usage.py                   # Приклад використання всіх семінарів
├── requirements.txt                   # Залежності проєкту
└── README_code.md                     # Цей файл
```

## Встановлення

```bash
pip install -r requirements.txt
```

## Швидкий старт

### Семінар 1: Виявлення інваріантів

```python
from seminar1_invariant_discovery import InvariantDiscovery
import pandas as pd

# Завантаження даних
data = pd.read_csv('metrics.csv')

# Створення об'єкта
discovery = InvariantDiscovery(
    variance_threshold=0.95,
    correlation_threshold=0.93,
    overhead=0.1
)

# Додавання груп метрик
discovery.add_feature_group('cpu_metrics', ['cpu_usage', 'cpu_freq', 'cpu_temp'])
discovery.add_feature_group('memory_metrics', ['ram_usage', 'swap_usage', 'cache_hits'])

# Навчання
invariants = discovery.fit(data)

# Детекція аномалій
results = discovery.detect_anomalies(new_data)
```

### Семінар 2: Діагностика через SHAP/LIME

```python
from seminar2_shap_lime import InvariantDiagnostics

# Створення діагностики
diagnostics = InvariantDiagnostics()

# SHAP пояснення
shap_result = diagnostics.diagnose_violation(
    invariant=invariant,
    data=anomaly_data,
    violation_row_idx=0,
    background_data=normal_data,
    method='shap'
)

# LIME пояснення
lime_result = diagnostics.diagnose_violation(
    invariant=invariant,
    data=anomaly_data,
    violation_row_idx=0,
    background_data=normal_data,
    method='lime'
)
```

### Семінар 3: Прескриптивний аналіз

```python
from seminar3_performance_doctor import PerformanceDoctor

# Створення PerformanceDoctor
doctor = PerformanceDoctor(
    actionable_features=['cpu_usage', 'ram_usage', 'thread_pool_size'],
    non_actionable_features=['timestamp', 'user_id'],
    n_counterfactuals=5
)

# Генерація рекомендації
prescription = doctor.prescribe_solution(
    invariant=violated_invariant,
    current_state=anomaly_data,
    row_idx=0,
    background_data=normal_data
)

# Виведення рекомендації
message = doctor.format_prescription_message(prescription)
print(message)
```

## Повний приклад

Запустіть `example_usage.py` для демонстрації всієї системи:

```bash
python example_usage.py
```

## Документація

Детальний методологічний та математичний опис кожного семінару:

- **[Семінар 1: Виявлення інваріантів](./07_seminar1_invariant_discovery.md)**
- **[Семінар 2: SHAP та LIME](./08_seminar2_shap_lime.md)**
- **[Семінар 3: Прескриптивний аналіз](./09_seminar3_performance_doctor.md)**

## Залежності

- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `scipy >= 1.7.0`
- `shap >= 0.41.0` (опціонально, є спрощена реалізація)
- `matplotlib >= 3.5.0` (для візуалізації)
- `seaborn >= 0.11.0` (для візуалізації)

## Примітки

- Якщо бібліотека `shap` не встановлена, використовується спрощена реалізація KernelSHAP
- Всі модулі повністю документовані з docstrings
- Код слідує PEP 8 стилю


