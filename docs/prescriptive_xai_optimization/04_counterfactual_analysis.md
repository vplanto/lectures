---
title: "04 Counterfactual Analysis"
type: lecture
module: Module 4
prerequisites: module 3
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# Контрфактуальний аналіз: Що, якби?

## Пролог: Мислення про альтернативні реальності

Уявіть: ваш сервер має latency 350ms (проблема). Ви знаєте, що `db_calls = 280` — це багато. Але що саме треба змінити, щоб latency повернулася до норми (< 200ms)?

**Це питання контрфактуального мислення:** "Що, якби `db_calls` було 150 замість 280? Яка була б latency?"

Контрфактуальний аналіз — це не просто пояснення ("чому так сталося"), а генерація **дійових рекомендацій** ("що треба змінити").

---

## Від пояснення до дії

### Еволюція XAI

1. **Descriptive (Описовий):** "Latency висока" — це факт.
2. **Diagnostic (Діагностичний):** "Latency висока, бо `db_calls` велике" — це пояснення (SHAP/LIME).
3. **Prescriptive (Прескриптивний):** "Зменш `db_calls` до 150, щоб latency стала < 200ms" — це дія.

**Контрфактуальний аналіз** — це перехід від Diagnostic до Prescriptive.

---

## Формальне визначення

### Контрфактуальний приклад

Нехай:
- $x$ — поточний стан системи (features)
- $y = f(x)$ — поточний прогноз (latency = 350ms)
- $y^*$ — бажаний результат (latency < 200ms)

**Контрфактуальний приклад** $x_{cf}$ — це такий стан, що:
1. $f(x_{cf}) \approx y^*$ (прогноз близький до бажаного)
2. $d(x, x_{cf})$ мінімальна (мінімальна зміна від поточного стану)
3. $x_{cf}$ реалістичний (можливий у реальному світі)

де $d$ — метрика відстані (наприклад, L1 або L2 норма).

### Математична формулювання

$$\min_{x_{cf}} \quad d(x, x_{cf}) + \lambda \cdot L(f(x_{cf}), y^*)$$

де:
- $d(x, x_{cf})$ — відстань між поточним та контрфактуальним станом
- $L(f(x_{cf}), y^*)$ — втрата між прогнозом та бажаним результатом
- $\lambda$ — гіперпараметр балансу

**Обмеження:**
- $x_{cf} \in \mathcal{X}_{valid}$ (реалістичні значення)
- Можливо, лише деякі ознаки можна змінювати (actionable features)

---

## Алгоритм DiCE: Diverse Counterfactual Explanations

DiCE (Mothilal et al., 2020) — популярна бібліотека для генерації контрфактуалів.

### Основні принципи

1. **Diversity:** Генерує кілька різних контрфактуалів (не один).
2. **Proximity:** Контрфактуали близькі до поточного стану.
3. **Sparsity:** Мінімальна кількість змінених ознак.

### Приклад використання DiCE

```python
import dice_ml
from dice_ml import Dice
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Підготовка даних
data = pd.DataFrame({
    'cpu_usage': [0.42, 0.58, 0.35, 0.72, 0.45],
    'db_calls': [125, 280, 95, 320, 150],
    'memory_usage': [0.68, 0.82, 0.55, 0.91, 0.70],
    'network_latency': [45, 52, 38, 65, 48],
    'latency': [150, 350, 120, 420, 180]
})

X = data.drop('latency', axis=1)
y = data['latency']

# Навчання моделі
model = RandomForestRegressor()
model.fit(X, y)

# Створення DiCE explainer
d = dice_ml.Data(data, continuous_features=X.columns.tolist(), outcome_name='latency')
m = dice_ml.Model(model=model, backend='sklearn')
explainer = Dice(d, m, method='random')

# Поточний стан (проблема)
query_instance = pd.DataFrame({
    'cpu_usage': [0.58],
    'db_calls': [280],
    'memory_usage': [0.82],
    'network_latency': [52]
})

# Генерація контрфактуалів
# Мета: latency < 200ms
counterfactuals = explainer.generate_counterfactuals(
    query_instance,
    total_CFs=5,  # 5 різних варіантів
    desired_range=[0, 200]  # Бажаний діапазон latency
)

# Візуалізація
counterfactuals.visualize_as_dataframe()
```

**Вихід:**

```
Поточний стан:
  cpu_usage: 0.58, db_calls: 280, memory_usage: 0.82, network_latency: 52
  Прогноз: 350ms

Контрфактуали:
CF1: cpu_usage: 0.58, db_calls: 150, memory_usage: 0.82, network_latency: 52
     Прогноз: 185ms ✓

CF2: cpu_usage: 0.45, db_calls: 200, memory_usage: 0.70, network_latency: 48
     Прогноз: 195ms ✓

CF3: cpu_usage: 0.50, db_calls: 180, memory_usage: 0.75, network_latency: 50
     Прогноз: 198ms ✓
```

---

## Власна реалізація контрфактуального пошуку

### Простий алгоритм: Gradient-based search

```python
import numpy as np
from scipy.optimize import minimize

def generate_counterfactual(model, instance, target_value, 
                           feature_bounds, actionable_features=None):
    """
    Генерація контрфактуального прикладу
    
    Args:
        model: навчена модель
        instance: поточний стан (1D array)
        target_value: бажане значення прогнозу
        feature_bounds: словник {feature_idx: (min, max)}
        actionable_features: список індексів ознак, які можна змінювати
    
    Returns:
        counterfactual: контрфактуальний приклад
    """
    if actionable_features is None:
        actionable_features = list(range(len(instance)))
    
    # Функція втрати
    def loss(cf_features):
        # Відновлюємо повний вектор (незмінні ознаки залишаються)
        cf_full = instance.copy()
        for i, feat_idx in enumerate(actionable_features):
            cf_full[feat_idx] = cf_features[i]
        
        # Прогноз
        prediction = model.predict([cf_full])[0]
        
        # Втрата: різниця між прогнозом та бажаним значенням
        prediction_loss = (prediction - target_value) ** 2
        
        # Втрата відстані: L1 норма
        distance_loss = np.sum(np.abs(cf_full - instance))
        
        return prediction_loss + 0.1 * distance_loss
    
    # Початкове наближення (поточний стан)
    x0 = np.array([instance[i] for i in actionable_features])
    
    # Обмеження (bounds)
    bounds = [feature_bounds[i] for i in actionable_features]
    
    # Оптимізація
    result = minimize(loss, x0, method='L-BFGS-B', bounds=bounds)
    
    # Відновлюємо повний вектор
    counterfactual = instance.copy()
    for i, feat_idx in enumerate(actionable_features):
        counterfactual[feat_idx] = result.x[i]
    
    return counterfactual, model.predict([counterfactual])[0]

# Приклад використання
instance = np.array([0.58, 280, 0.82, 52])  # Поточний стан
target_latency = 200  # Бажана latency

feature_bounds = {
    0: (0.0, 1.0),      # cpu_usage
    1: (0, 500),        # db_calls
    2: (0.0, 1.0),      # memory_usage
    3: (0, 100)         # network_latency
}

# Можна змінювати лише db_calls та cpu_usage
actionable = [0, 1]  # cpu_usage, db_calls

cf, predicted = generate_counterfactual(
    model, instance, target_latency, feature_bounds, actionable
)

print(f"Поточний стан: {instance}")
print(f"Прогноз: {model.predict([instance])[0]:.1f}ms")
print(f"\nКонтрфактуальний стан: {cf}")
print(f"Прогноз: {predicted:.1f}ms")
print(f"\nЗміни:")
for i, (old, new) in enumerate(zip(instance, cf)):
    if abs(old - new) > 1e-6:
        print(f"  Feature {i}: {old:.2f} → {new:.2f} (Δ {new-old:+.2f})")
```

---

## Контрфактуали для класифікації

Для класифікації мета — змінити клас.

```python
def generate_counterfactual_classification(model, instance, 
                                          target_class, feature_bounds):
    """
    Генерація контрфактуального прикладу для класифікації
    """
    def loss(cf_features):
        cf_full = instance.copy()
        for i, val in enumerate(cf_features):
            cf_full[i] = val
        
        # Прогноз ймовірностей
        proba = model.predict_proba([cf_full])[0]
        
        # Втрата: максимізуємо ймовірність target_class
        prediction_loss = 1 - proba[target_class]
        
        # Втрата відстані
        distance_loss = np.sum(np.abs(cf_full - instance))
        
        return prediction_loss + 0.1 * distance_loss
    
    x0 = instance.copy()
    bounds = [feature_bounds[i] for i in range(len(instance))]
    
    result = minimize(loss, x0, method='L-BFGS-B', bounds=bounds)
    
    return result.x, model.predict_proba([result.x])[0]

# Приклад: змінити клас з "Аномалія" на "Норма"
instance = np.array([0.58, 280, 0.82, 52])
target_class = 0  # "Норма"

cf, proba = generate_counterfactual_classification(
    model, instance, target_class, feature_bounds
)

print(f"Поточний стан: клас {model.predict([instance])[0]} "
      f"(ймовірність: {model.predict_proba([instance])[0]})")
print(f"Контрфактуальний стан: клас {model.predict([cf])[0]} "
      f"(ймовірність: {proba})")
```

---

## Інтеграція з Comparator Engine та SHAP

Повний цикл: Detection → Diagnosis → Prescription

```python
from comparator_engine import ComparatorEngine, SystemProfile

def full_diagnostic_pipeline(baseline_profile: SystemProfile,
                            target_profile: SystemProfile,
                            model, explainer_shap, explainer_dice):
    """
    Повний пайплайн: від виявлення до рекомендацій
    """
    # 1. Detection
    comparator = ComparatorEngine(THRESHOLDS)
    comparison = comparator.compare(baseline_profile, target_profile)
    
    if not comparison['violations']:
        return {"status": "no_issues"}
    
    # 2. Diagnosis (SHAP)
    baseline_features = pd.DataFrame([baseline_profile.metrics])
    target_features = pd.DataFrame([target_profile.metrics])
    
    shap_values = explainer_shap.shap_values(target_features)
    
    # Знаходимо найбільший внесок
    top_contributor_idx = np.argmax(np.abs(shap_values[0]))
    top_contributor = target_features.columns[top_contributor_idx]
    
    # 3. Prescription (Counterfactual)
    # Мета: повернути latency до baseline значення
    target_latency = baseline_features['request_latency_p50'].iloc[0]
    
    current_state = target_features.iloc[0].values
    counterfactuals = explainer_dice.generate_counterfactuals(
        target_features,
        total_CFs=3,
        desired_range=[target_latency * 0.9, target_latency * 1.1]
    )
    
    # Генерація рекомендацій
    recommendations = []
    for cf in counterfactuals.cf_examples_list[0].final_cfs_df.iterrows():
        changes = {}
        for col in target_features.columns:
            old_val = current_state[target_features.columns.get_loc(col)]
            new_val = cf[1][col]
            if abs(old_val - new_val) > 1e-6:
                changes[col] = {
                    'old': old_val,
                    'new': new_val,
                    'delta': new_val - old_val
                }
        recommendations.append({
            'changes': changes,
            'predicted_latency': model.predict([cf[1].values])[0]
        })
    
    return {
        'detection': {
            'violations': comparison['violations'],
            'top_contributor': top_contributor
        },
        'diagnosis': {
            'shap_contributions': dict(zip(
                target_features.columns,
                shap_values[0]
            ))
        },
        'prescription': {
            'recommendations': recommendations
        }
    }

# Використання
result = full_diagnostic_pipeline(
    baseline_profile,
    target_profile,
    model,
    shap_explainer,
    dice_explainer
)

print("ДІАГНОСТИКА:")
print(f"Проблема: {result['detection']['top_contributor']}")
print(f"\nПояснення (SHAP):")
for feat, contrib in result['diagnosis']['shap_contributions'].items():
    print(f"  {feat}: {contrib:+.1f}ms")

print(f"\nРЕКОМЕНДАЦІЇ:")
for i, rec in enumerate(result['prescription']['recommendations'], 1):
    print(f"\nВаріант {i}:")
    for feat, change in rec['changes'].items():
        print(f"  {feat}: {change['old']:.2f} → {change['new']:.2f} "
              f"(Δ {change['delta']:+.2f})")
    print(f"  Прогнозована latency: {rec['predicted_latency']:.1f}ms")
```

---

## Обмеження та виклики

### 1. Реалістичність

Контрфактуали можуть бути математично валідними, але фізично неможливими.

**Приклад:** "Зменш CPU usage до 0.1" — можливо, але якщо система завантажена, це неможливо без зміни навантаження.

**Рішення:** Додати обмеження на взаємозв'язки ознак (causal constraints).

### 2. Actionable vs Non-actionable

Не всі ознаки можна змінювати напряму.

- **Actionable:** `db_calls` (можна оптимізувати запити), `memory_limit` (можна збільшити)
- **Non-actionable:** `timestamp`, `user_id`

**Рішення:** Фільтрувати лише actionable features.

### 3. Багато варіантів

Може бути багато різних контрфактуалів. Який обрати?

**Рішення:** DiCE генерує кілька варіантів. Користувач обирає найбільш реалістичний.

---

## Висновок: Від "Чому?" до "Що робити?"

Контрфактуальний аналіз завершує цикл XAI:

1. **Detection (Comparator):** Що змінилося?
2. **Diagnosis (SHAP/LIME):** Чому це сталося?
3. **Prescription (Counterfactual):** Що треба змінити?

Це перетворює "чорну скриньку" на **автоматичного "лікаря" системи**, що не тільки діагностує проблему, але й пропонує конкретні дії для виправлення.

У [Блоці 4](./05_building_performance_doctor.md) ми інтегруємо всі компоненти в повноцінну систему "Performance Doctor".

---

## Домашнє завдання

1. Реалізуйте власний алгоритм пошуку контрфактуалів з підтримкою causal constraints (наприклад, якщо `db_calls` зменшується, `cpu_usage` теж може зменшитися).
2. Порівняйте DiCE та власну реалізацію: чи дають вони подібні результати?
3. Створіть систему ранжування контрфактуалів за "реалістичністю" (наприклад, на основі історичних даних).


