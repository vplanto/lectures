---
title: "Ефект Розумного Ганса: Коли AI обдурює нас"
layout: default
nav_order: 0
---

# Ефект Розумного Ганса: Коли AI обдурює нас

## Пролог: Конь, що обдурив науку

1904 рік. Німеччина. Вільгельм фон Остен демонструє свого коня Ганса, який, здається, розуміє математику. Запитуєш: "Скільки буде 3 + 5?" — Ганс стукає копитом 8 разів. Запитуєш: "Яка дата сьогодні?" — стукає правильно. Вчені в захваті: конь має інтелект!

Але психолог Оскар Пфунгст розкриває обман. Ганс не рахував. Він читав мікро-вирази обличчя глядачів: коли досягав правильної відповіді, люди непомітно розслаблялися. Конь зупинявся саме в цей момент.

**Це і є ефект Розумного Ганса:** модель досягає високої точності, але не тому, що розуміє задачу, а тому, що використовує приховані підказки в даних.

---

## Чому це важливо для нас?

У 2017 році дослідники з Університету Вашингтона навчили нейромережу розпізнавати вовків та хаскі. Модель досягла 90% точності. Але коли її протестували на нових зображеннях, виявилося: вона не розпізнавала вовків. Вона розпізнавала **сніг на фоні**.

Всі зображення вовків у тренувальному наборі були на сніжному тлі. Всі хаскі — на зеленому. Модель вивчила кореляцію "сніг = вовк", а не фізичні ознаки тварин.

```python
# Псевдокод проблеми
def train_wolf_classifier(images):
    # Модель бачить:
    # Вовки: [сніг, сніг, сніг, ...]
    # Хаскі: [трава, трава, трава, ...]
    
    # Модель вчить:
    # if background == "snow":
    #     return "wolf"
    # else:
    #     return "husky"
    
    # Реальність: модель не розуміє вовків!
    pass
```

---

## Архітектурна проблема: "Чорна скринька"

### Патерн: Black Box Anti-Pattern

У класичній архітектурі ML-систем ми маємо:

```
[Вхідні дані] → [Чорна скринька (ML Model)] → [Прогноз]
```

**Проблема:** Ми не знаємо, *чому* модель прийняла рішення. Це створює три критичні ризики:

1. **Ризик довіри:** Коли модель помиляється, ми не можемо виправити помилку.
2. **Ризик регуляції:** GDPR, EU AI Act вимагають пояснення автоматичних рішень.
3. **Ризик бізнесу:** Неможливо перевірити, чи модель використовує правильні ознаки.

### Рішення: Explainable AI (XAI)

XAI — це не одна технологія, а архітектурний підхід, що додає **шар інтерпретації** до моделі:

```
[Вхідні дані] → [ML Model] → [Прогноз]
                      ↓
              [XAI Explainer] → [Пояснення]
```

---

## Вступ до XAI: Три рівні пояснень

### 1. Глобальна інтерпретація (Global Interpretability)

**Питання:** "Які ознаки загалом важливі для моделі?"

**Відповідь:** Feature Importance, Permutation Importance.

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Приклад: модель прогнозує latency сервера
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Глобальна важливість ознак
importances = model.feature_importances_
feature_names = ['CPU_usage', 'DB_calls', 'Network_latency', 'Memory_usage']

for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")
```

**Висновок:** "CPU_usage" — найважливіша ознака (0.45), "DB_calls" — друга (0.32).

### 2. Локальна інтерпретація (Local Interpretability)

**Питання:** "Чому для *цього конкретного* запиту модель передбачила latency = 350ms?"

**Відповідь:** SHAP Values, LIME.

```python
import shap

# Локальне пояснення для одного прикладу
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[0:1])

# Візуалізація
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=feature_names
))
```

**Висновок:** "Для цього запиту DB_calls додав +120ms, а CPU_usage зменшив на -30ms. Чиста зміна: +90ms від базового значення."

### 3. Контрфактуальне пояснення (Counterfactual)

**Питання:** "Що треба змінити, щоб отримати інший результат?"

**Відповідь:** DiCE, What-If Tool.

```python
# Псевдокод контрфактуального аналізу
current_state = {
    'CPU_usage': 0.85,
    'DB_calls': 150,
    'Memory_usage': 0.70
}
predicted_latency = model.predict([current_state])  # 350ms

# Що, якби зменшити DB_calls до 100?
counterfactual = current_state.copy()
counterfactual['DB_calls'] = 100
new_latency = model.predict([counterfactual])  # 280ms

print(f"Зменшення DB_calls на 50 зменшує latency на {350-280}ms")
```

---

## Практичний приклад: LIME для класифікації

Розглянемо, як LIME розкриває "обман" моделі:

```python
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier

# Модель класифікує: "Аномалія" vs "Норма"
model = RandomForestClassifier()
model.fit(X_train, y_train)

# LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=['Норма', 'Аномалія'],
    mode='classification'
)

# Пояснення для одного прикладу
explanation = explainer.explain_instance(
    X_test.iloc[0].values,
    model.predict_proba,
    num_features=5
)

# Вивід
explanation.show_in_notebook(show_table=True)
```

**Що показує LIME:**

```
Прогноз: Аномалія (ймовірність: 0.87)

Ознаки, що підтримують "Аномалія":
- DB_calls > 200: +0.35
- CPU_usage > 0.90: +0.28
- Memory_usage > 0.85: +0.15

Ознаки, що підтримують "Норма":
- Network_latency < 50ms: -0.12
```

---

## Висновок: Від "Чорної скриньки" до прозорості

Ефект Розумного Ганса нагадує нам: **висока точність не означає розуміння**. XAI — це не просто "nice to have", а критичний компонент production ML-систем.

### Ключові висновки:

1. **Кореляція ≠ Причинність:** Модель може вивчити спускні ознаки (як "сніг" для вовків).
2. **Глобальна vs Локальна:** Різні методи дають різні рівні розуміння.
3. **Архітектурний підхід:** XAI — це не одна бібліотека, а шаблон проектування системи.

### Наступний крок:

У [Блоці 1](./01_comparative_architecture.md) ми розберемо, як архітектурно організувати порівняльний аналіз для виявлення деградації продуктивності — фундамент для подальшого застосування XAI.

---

## Додаткові матеріали

- **Стаття:** ["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) (LIME)
- **Книга:** "Interpretable Machine Learning" by Christoph Molnar
- **Відео:** Veritasium — ["The Most Misunderstood Concept in Physics"](https://www.youtube.com/watch?v=Y-vw8q0QJ8I) (про кореляцію та причинність)


