---
title: "Значення Шеплі: Теорія ігор для пояснення AI"
layout: default
nav_order: 2
---

# Значення Шеплі: Теорія ігор для пояснення AI

## Пролог: Справедливий розподіл виграшу

1953 рік. Ллойд Шеплі, математик з Принстона, розв'язує задачу: як чесно розділити виграш між гравцями в кооперативній грі?

**Приклад:** Троє гравців виграли $100. Гравець A сам виграє $30. Гравець B сам виграє $50. Гравець C сам виграє $0. Але разом вони виграють $100. Як розділити справедливо?

Шеплі запропонував формулу, що враховує **маржинальний внесок** кожного гравця в усі можливі коаліції. Ця ідея стала основою для пояснення машинного навчання.

---

## Від теорії ігор до машинного навчання

### Аналогія: Гравці = Ознаки

У контексті ML:

- **Гравці** = ознаки (features): `CPU_usage`, `DB_calls`, `Memory_usage`
- **Виграш** = прогноз моделі: `predicted_latency = 350ms`
- **Коаліції** = підмножини ознак: `{CPU_usage}`, `{CPU_usage, DB_calls}`, тощо

**Питання:** Який внесок кожної ознаки в фінальний прогноз?

### Формальне визначення

Нехай $f$ — це наша модель, $x = (x_1, x_2, \ldots, x_n)$ — вектор ознак, $S \subseteq \{1, 2, \ldots, n\}$ — підмножина індексів.

**Значення Шеплі** для ознаки $i$:

$$\phi_i(f, x) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} \left[ f(x_S \cup \{i\}) - f(x_S) \right]$$

де:
- $N = \{1, 2, \ldots, n\}$ — множина всіх ознак
- $x_S$ — вектор, де ознаки з $S$ мають свої значення, а інші — базові (наприклад, середні)
- $f(x_S \cup \{i\}) - f(x_S)$ — маржинальний внесок ознаки $i$ до коаліції $S$

**Інтуїція:** Значення Шеплі — це середньозважений маржинальний внесок ознаки по всіх можливих коаліціях.

---

## SHAP: SHapley Additive exPlanations

SHAP — це бібліотека, що реалізує значення Шеплі для ML-моделей.

### Властивості SHAP

1. **Ефективність (Efficiency):** $\sum_{i=1}^n \phi_i = f(x) - f(\emptyset)$
   - Сума всіх SHAP-значень дорівнює різниці між прогнозом та базовим значенням.

2. **Симетрія (Symmetry):** Якщо дві ознаки мають однаковий внесок у всіх коаліціях, їх SHAP-значення рівні.

3. **Нульовий внесок (Dummy):** Якщо ознака не впливає на прогноз, її SHAP-значення = 0.

4. **Адитивність (Additivity):** Для ансамблів моделей SHAP-значення адитивні.

---

## Практичний приклад: Пояснення прогнозу latency

### Підготовка даних

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap

# Приклад даних: метрики сервера
data = {
    'cpu_usage': [0.42, 0.58, 0.35, 0.72, 0.45],
    'db_calls': [125, 280, 95, 320, 150],
    'memory_usage': [0.68, 0.82, 0.55, 0.91, 0.70],
    'network_latency': [45, 52, 38, 65, 48],
    'latency': [150, 350, 120, 420, 180]  # Target
}

df = pd.DataFrame(data)
X = df.drop('latency', axis=1)
y = df['latency']

# Навчання моделі
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
```

### Глобальна інтерпретація: Важливість ознак

```python
# Tree SHAP explainer (швидкий для tree-based моделей)
explainer = shap.TreeExplainer(model)

# Обчислення SHAP-значень для всього набору
shap_values = explainer.shap_values(X)

# Візуалізація глобальної важливості
shap.summary_plot(shap_values, X, plot_type="bar")
```

**Висновок:** `db_calls` — найважливіша ознака, далі `cpu_usage`, `memory_usage`, `network_latency`.

### Локальна інтерпретація: Пояснення одного прогнозу

```python
# Виберемо приклад з високою latency
sample_idx = 1  # latency = 350ms
sample = X.iloc[sample_idx:sample_idx+1]

# SHAP-значення для цього прикладу
shap_values_sample = explainer.shap_values(sample)

# Базове значення (середній прогноз)
base_value = explainer.expected_value

print(f"Базове значення (середній прогноз): {base_value:.1f}ms")
print(f"Прогноз для цього прикладу: {model.predict(sample)[0]:.1f}ms")
print(f"Різниця: {model.predict(sample)[0] - base_value:.1f}ms")
print("\nВнесок кожної ознаки:")
for i, feature in enumerate(X.columns):
    print(f"  {feature}: {shap_values_sample[0][i]:+.1f}ms")
```

**Вихід:**

```
Базове значення (середній прогноз): 244.0ms
Прогноз для цього прикладу: 350.0ms
Різниця: 106.0ms

Внесок кожної ознаки:
  cpu_usage: +28.5ms
  db_calls: +95.2ms
  memory_usage: +12.3ms
  network_latency: -30.0ms
```

**Інтерпретація:** 
- `db_calls` додає +95.2ms (головна причина високої latency)
- `cpu_usage` додає +28.5ms
- `network_latency` зменшує на -30.0ms (низька мережева затримка частково компенсує)
- **Сума:** 28.5 + 95.2 + 12.3 - 30.0 = 106.0ms (збігається з різницею!)

### Візуалізація: Waterfall plot

```python
# Waterfall plot для одного прикладу
shap.waterfall_plot(shap.Explanation(
    values=shap_values_sample[0],
    base_values=base_value,
    data=sample.iloc[0],
    feature_names=X.columns
))
```

Waterfall plot показує, як кожна ознака "штовхає" прогноз від базового значення до фінального.

---

## SHAP для різних типів моделей

### 1. Tree SHAP (для Random Forest, XGBoost, LightGBM)

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

**Переваги:** Швидкий, точний для tree-based моделей.

### 2. Kernel SHAP (універсальний)

```python
# Для будь-якої моделі (повільніший, але універсальний)
explainer = shap.KernelExplainer(model.predict, X.iloc[:100])  # Background data
shap_values = explainer.shap_values(X.iloc[0:1])  # Для одного прикладу
```

**Переваги:** Працює з будь-якою моделлю (нейромережі, SVM, тощо).

**Недоліки:** Повільний, потребує background dataset.

### 3. Deep SHAP (для нейромереж)

```python
import tensorflow as tf

# Для TensorFlow/Keras моделей
explainer = shap.DeepExplainer(model, X_background)
shap_values = explainer.shap_values(X_sample)
```

---

## Інтеграція з Comparator Engine

Тепер об'єднаємо Comparator Engine з SHAP для повного циклу: Detection → Diagnosis.

```python
from comparator_engine import ComparatorEngine, SystemProfile

def diagnose_degradation(baseline_profile: SystemProfile,
                        target_profile: SystemProfile,
                        model, explainer):
    """
    Діагностика деградації: Detection + Diagnosis
    """
    # 1. Detection: Порівняння профілів
    comparator = ComparatorEngine(THRESHOLDS)
    comparison = comparator.compare(baseline_profile, target_profile)
    
    # Знаходимо метрики з найбільшою деградацією
    top_violations = sorted(
        comparison['violations'],
        key=lambda v: abs(v['delta'].rel_diff),
        reverse=True
    )[:3]
    
    # 2. Diagnosis: SHAP для пояснення
    # Конвертуємо профілі в feature vectors
    baseline_features = pd.DataFrame([baseline_profile.metrics])
    target_features = pd.DataFrame([target_profile.metrics])
    
    # SHAP-значення для baseline та target
    shap_baseline = explainer.shap_values(baseline_features)
    shap_target = explainer.shap_values(target_features)
    
    # Різниця SHAP-значень показує, які ознаки "викликали" деградацію
    shap_diff = shap_target[0] - shap_baseline[0]
    
    diagnosis = {
        'detected_violations': top_violations,
        'shap_contributions': {
            feature: contribution 
            for feature, contribution in zip(
                baseline_features.columns, shap_diff
            )
        }
    }
    
    return diagnosis

# Використання
diagnosis = diagnose_degradation(
    baseline_profile,
    target_profile,
    model,
    explainer
)

print("Діагностика деградації:")
print(f"Найбільша деградація: {diagnosis['detected_violations'][0]['metric']}")
print("\nSHAP-внесок у деградацію:")
for feature, contribution in sorted(
    diagnosis['shap_contributions'].items(),
    key=lambda x: abs(x[1]),
    reverse=True
):
    print(f"  {feature}: {contribution:+.1f}ms")
```

---

## Математична інтуїція: Чому SHAP справедливий?

### Приклад з 2 ознаками

Нехай модель: $f(x_1, x_2) = 2x_1 + 3x_2 + 10$

Для прикладу $(x_1=5, x_2=4)$:
- $f(\emptyset) = 10$ (базове значення)
- $f(\{1\}) = 2 \cdot 5 + 10 = 20$ (лише $x_1$)
- $f(\{2\}) = 3 \cdot 4 + 10 = 22$ (лише $x_2$)
- $f(\{1,2\}) = 2 \cdot 5 + 3 \cdot 4 + 10 = 32$ (обидві)

**Обчислення SHAP:**

Для $x_1$:
- Коаліція $\emptyset$: $f(\{1\}) - f(\emptyset) = 20 - 10 = 10$, вага: $\frac{0! \cdot 1!}{2!} = \frac{1}{2}$
- Коаліція $\{2\}$: $f(\{1,2\}) - f(\{2\}) = 32 - 22 = 10$, вага: $\frac{1! \cdot 0!}{2!} = \frac{1}{2}$

$\phi_1 = \frac{1}{2} \cdot 10 + \frac{1}{2} \cdot 10 = 10$

Для $x_2$:
- Коаліція $\emptyset$: $f(\{2\}) - f(\emptyset) = 22 - 10 = 12$, вага: $\frac{1}{2}$
- Коаліція $\{1\}$: $f(\{1,2\}) - f(\{1\}) = 32 - 20 = 12$, вага: $\frac{1}{2}$

$\phi_2 = \frac{1}{2} \cdot 12 + \frac{1}{2} \cdot 12 = 12$

**Перевірка:** $\phi_1 + \phi_2 = 10 + 12 = 22 = f(5,4) - f(\emptyset) = 32 - 10$ ✓

---

## Висновок: Від "Що?" до "Чому?"

SHAP дає нам математично обґрунтовану відповідь на питання "Чому модель передбачила це значення?". Це критично для:

1. **Довіри:** Можемо перевірити, чи модель використовує правильні ознаки.
2. **Діагностики:** Розуміємо, які фактори викликали проблему.
3. **Дотримання регуляцій:** GDPR вимагає пояснення автоматичних рішень.

У [Блоці 3](./03_lime_local_approximation.md) ми розглянемо альтернативний підхід — LIME, який працює через локальну апроксимацію.

---

## Домашнє завдання

1. Реалізуйте функцію `compute_shapley_value_manual` для обчислення SHAP-значень "вручну" (для невеликої кількості ознак).
2. Порівняйте Tree SHAP та Kernel SHAP на одній моделі: чи однакові результати?
3. Створіть інтерактивну dashboard з SHAP plots для аналізу деградації продуктивності.


