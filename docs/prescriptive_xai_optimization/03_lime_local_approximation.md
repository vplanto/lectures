---
title: "LIME: Локальна апроксимація для пояснення AI"
layout: default
nav_order: 3
---

# LIME: Локальна апроксимація для пояснення AI

## Пролог: Апроксимація складного простого

Уявіть: ви стоїте біля складного гірського ландшафту. Щоб зрозуміти форму гори, ви не намагаєтеся описати всю гору одразу. Замість цього ви дивитеся на невелику ділянку навколо себе та апроксимуєте її плоскою поверхнею (дотичною площиною).

**Це і є ідея LIME:** замість пояснення всієї моделі глобально, ми будуємо просту інтерпретовану модель *локально* навколо точки інтересу.

---

## Філософія LIME: "Місцева простота"

### Проблема глобального пояснення

Складні моделі (нейромережі, ансамбли) мають нелінійні залежності, взаємодії ознак, складні градієнти. Глобально пояснити таку модель важко.

**Рішення LIME:** Якщо глобально пояснити важко, пояснимо **локально**.

### Математична інтуїція

Нехай $f$ — наша складна модель, $x$ — точка інтересу.

LIME шукає просту модель $g$ (наприклад, лінійну), яка **апроксимує** $f$ навколо $x$:

$$g(z) = \beta_0 + \sum_{i=1}^n \beta_i z_i$$

де $z$ — це спрощена версія $x$ (наприклад, бінарний вектор: ознака присутня/відсутня).

**Критерій оптимізації:**

$$\min_{g \in G} L(f, g, \pi_x) + \Omega(g)$$

де:
- $L(f, g, \pi_x)$ — втрата між $f$ та $g$ на прикладах, зважених функцією близькості $\pi_x$
- $\Omega(g)$ — регуляризація (наприклад, кількість ненульових коефіцієнтів)
- $G$ — сімейство простих моделей (лінійні, дерева рішень)

---

## Алгоритм LIME

### Крок 1: Генерація зразків навколо точки інтересу

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def generate_perturbed_samples(instance, num_samples=5000):
    """
    Генерація зразків навколо instance
    
    Args:
        instance: оригінальний приклад (1D array)
        num_samples: кількість зразків для генерації
    
    Returns:
        perturbed_samples: згенеровані зразки
        weights: ваги близькості до оригіналу
    """
    n_features = len(instance)
    
    # Генерація випадкових бінарних векторів (0/1)
    # 1 = ознака присутня (беремо значення з instance)
    # 0 = ознака відсутня (беремо середнє значення)
    perturbed_samples = np.random.binomial(1, 0.5, (num_samples, n_features))
    
    # Замінюємо 1 на значення з instance, 0 на середнє
    mean_values = np.mean(instance)  # Спрощено: можна використовувати background data
    samples = perturbed_samples.copy().astype(float)
    samples[perturbed_samples == 1] = instance[perturbed_samples == 1]
    samples[perturbed_samples == 0] = mean_values
    
    # Обчислення ваг близькості (експоненційне ядро)
    distances = np.sum(perturbed_samples != 1, axis=1)  # Кількість змінених ознак
    weights = np.exp(-distances / np.sqrt(n_features))  # Чим ближче, тим більша вага
    
    return samples, weights
```

### Крок 2: Отримання прогнозів складної моделі

```python
def get_predictions(model, samples):
    """Отримання прогнозів від складної моделі"""
    return model.predict(samples)
```

### Крок 3: Навчання простої моделі

```python
def train_lime_explainer(model, instance, num_samples=5000):
    """
    Навчання LIME explainer для одного прикладу
    
    Returns:
        explanation: словник з коефіцієнтами та інтерпретацією
    """
    # Генерація зразків
    samples, weights = generate_perturbed_samples(instance, num_samples)
    
    # Прогнози від складної моделі
    predictions = get_predictions(model, samples)
    
    # Навчання лінійної моделі з вагами
    # Використовуємо Ridge для стабільності
    explainer_model = Ridge(alpha=1.0)
    explainer_model.fit(samples, predictions, sample_weight=weights)
    
    # Коефіцієнти = внесок кожної ознаки
    feature_contributions = explainer_model.coef_
    intercept = explainer_model.intercept_
    
    return {
        'coefficients': feature_contributions,
        'intercept': intercept,
        'feature_names': [f'feature_{i}' for i in range(len(instance))]
    }
```

---

## Практичний приклад: LIME для класифікації аномалій

### Підготовка даних та моделі

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

# Дані: метрики сервера
data = {
    'cpu_usage': [0.42, 0.58, 0.35, 0.72, 0.45, 0.88, 0.25],
    'db_calls': [125, 280, 95, 320, 150, 450, 80],
    'memory_usage': [0.68, 0.82, 0.55, 0.91, 0.70, 0.95, 0.50],
    'network_latency': [45, 52, 38, 65, 48, 75, 35],
    'is_anomaly': [0, 1, 0, 1, 0, 1, 0]  # 1 = аномалія
}

df = pd.DataFrame(data)
X = df.drop('is_anomaly', axis=1)
y = df['is_anomaly']

# Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Навчання моделі
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

### Створення LIME explainer

```python
# Створення LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['Норма', 'Аномалія'],
    mode='classification'
)
```

### Пояснення для одного прикладу

```python
# Виберемо приклад з аномалією
sample_idx = 1  # Аномалія
instance = X_test.iloc[sample_idx].values

# Генерація пояснення
explanation = explainer.explain_instance(
    instance,
    model.predict_proba,
    num_features=len(X.columns),
    top_labels=1
)

# Вивід результату
print("Прогноз моделі:")
proba = model.predict_proba([instance])[0]
print(f"  Норма: {proba[0]:.3f}")
print(f"  Аномалія: {proba[1]:.3f}")
print(f"  Клас: {'Аномалія' if model.predict([instance])[0] == 1 else 'Норма'}")

print("\nПояснення (LIME):")
print("Ознаки, що підтримують 'Аномалія':")
for feature, weight in explanation.as_list(label=1):
    if weight > 0:
        print(f"  {feature}: +{weight:.3f}")

print("\nОзнаки, що підтримують 'Норма':")
for feature, weight in explanation.as_list(label=1):
    if weight < 0:
        print(f"  {feature}: {weight:.3f}")
```

**Вихід:**

```
Прогноз моделі:
  Норма: 0.123
  Аномалія: 0.877
  Клас: Аномалія

Пояснення (LIME):
Ознаки, що підтримують 'Аномалія':
  db_calls > 200.00: +0.452
  cpu_usage > 0.55: +0.321
  memory_usage > 0.80: +0.198

Ознаки, що підтримують 'Норма':
  network_latency < 50.00: -0.087
```

### Візуалізація

```python
# Візуалізація в Jupyter notebook
explanation.show_in_notebook(show_table=True)
```

---

## LIME для тексту та зображень

### LIME для тексту

```python
from lime import lime_text
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Приклад: класифікація тональності тексту
texts = ["Це чудовий продукт!", "Дуже погана якість.", ...]
labels = [1, 0, ...]  # 1 = позитивний, 0 = негативний

# Модель
vectorizer = TfidfVectorizer()
classifier = LogisticRegression()
pipeline = make_pipeline(vectorizer, classifier)
pipeline.fit(texts, labels)

# LIME explainer для тексту
explainer = lime_text.LimeTextExplainer(class_names=['Негативний', 'Позитивний'])

# Пояснення
text_instance = "Продукт непоганий, але доставка повільна"
explanation = explainer.explain_instance(
    text_instance,
    pipeline.predict_proba,
    num_features=10
)

explanation.show_in_notebook()
```

### LIME для зображень

```python
from lime import lime_image
from skimage.segmentation import quickshift

# LIME для зображень використовує сегментацію
explainer = lime_image.LimeImageExplainer()

# Сегментація зображення
segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)

# Пояснення
explanation = explainer.explain_instance(
    image,
    model.predict_proba,
    segmentation_fn=quickshift,
    top_labels=5,
    hide_color=0,
    num_samples=1000
)
```

---

## Порівняння LIME та SHAP

| Аспект | LIME | SHAP |
|--------|------|------|
| **Підхід** | Локальна апроксимація | Теорія ігор (Shapley Values) |
| **Швидкість** | Повільніший (потрібна генерація зразків) | Швидший для tree-based моделей |
| **Точність** | Апроксимація (може бути неточною) | Точний (для Tree SHAP) |
| **Інтерпретація** | Коефіцієнти лінійної моделі | Маржинальний внесок |
| **Застосування** | Будь-яка модель | Залежить від типу (Tree/Kernel/Deep) |
| **Глобальна інтерпретація** | Ні (лише локальна) | Так (можна агрегувати) |

### Коли використовувати LIME?

1. **Складні моделі без SHAP-підтримки:** нейромережі, кастомні моделі
2. **Текст та зображення:** LIME має спеціалізовані версії
3. **Швидкий прототип:** легше інтегрувати для будь-якої моделі

### Коли використовувати SHAP?

1. **Tree-based моделі:** Tree SHAP швидкий і точний
2. **Глобальна інтерпретація:** потрібно зрозуміти модель загалом
3. **Математична обґрунтованість:** потрібні гарантії (властивості Shapley)

---

## Інтеграція LIME з Comparator Engine

```python
from comparator_engine import ComparatorEngine, SystemProfile

def diagnose_with_lime(baseline_profile: SystemProfile,
                      target_profile: SystemProfile,
                      model, explainer):
    """
    Діагностика з використанням LIME
    """
    # 1. Detection
    comparator = ComparatorEngine(THRESHOLDS)
    comparison = comparator.compare(baseline_profile, target_profile)
    
    # 2. Diagnosis з LIME
    baseline_features = pd.DataFrame([baseline_profile.metrics])
    target_features = pd.DataFrame([target_profile.metrics])
    
    # Пояснення для baseline та target
    exp_baseline = explainer.explain_instance(
        baseline_features.iloc[0].values,
        model.predict_proba,
        num_features=len(baseline_features.columns)
    )
    
    exp_target = explainer.explain_instance(
        target_features.iloc[0].values,
        model.predict_proba,
        num_features=len(target_features.columns)
    )
    
    # Порівняння внесків
    baseline_contrib = {feat: weight for feat, weight in exp_baseline.as_list()}
    target_contrib = {feat: weight for feat, weight in exp_target.as_list()}
    
    # Різниця показує, що змінилося
    diff_contrib = {
        feat: target_contrib.get(feat, 0) - baseline_contrib.get(feat, 0)
        for feat in set(baseline_contrib.keys()) | set(target_contrib.keys())
    }
    
    return {
        'violations': comparison['violations'],
        'lime_contributions': diff_contrib
    }
```

---

## Обмеження LIME

### 1. Нестабільність

LIME використовує випадкову генерацію зразків. При повторному запуску результати можуть трохи відрізнятися.

**Рішення:** Середнє значення по кількох запусках.

```python
def stable_lime_explanation(explainer, instance, model, n_runs=10):
    """Стабілізація LIME через багаторазові запуски"""
    all_explanations = []
    for _ in range(n_runs):
        exp = explainer.explain_instance(instance, model.predict_proba)
        all_explanations.append(dict(exp.as_list()))
    
    # Середнє значення
    avg_explanation = {}
    for feat in set().union(*all_explanations):
        avg_explanation[feat] = np.mean([exp.get(feat, 0) for exp in all_explanations])
    
    return avg_explanation
```

### 2. Вибір функції близькості

Функція $\pi_x$ (ваги близькості) критична. Неправильний вибір може дати неправильні пояснення.

### 3. Локальність

LIME пояснює лише локально. Глобальна поведінка моделі може відрізнятися.

---

## Висновок: Локальна простота як сила

LIME показує, що для пояснення складної моделі не обов'язково розуміти її глобально. Достатньо побудувати просту апроксимацію навколо точки інтересу.

**Ключові висновки:**

1. **Локальна простота:** Складні моделі можна пояснити просто локально.
2. **Універсальність:** LIME працює з будь-якою моделлю.
3. **Інтуїтивність:** Лінійні коефіцієнти легко інтерпретувати.

У [Блоці 3 (частина 2)](./04_counterfactual_analysis.md) ми перейдемо від пояснення до дії: як генерувати рекомендації для виправлення проблем.

---

## Домашнє завдання

1. Реалізуйте власний LIME explainer для регресії (без використання бібліотеки `lime`).
2. Порівняйте LIME та SHAP на одній моделі: чи дають вони подібні результати?
3. Дослідіть вплив параметра `num_samples` на стабільність пояснень LIME.


