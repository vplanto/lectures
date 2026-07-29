---
title: "10 Seminar Metrics Extreme Imbalance"
type: seminar
module: Семінар
prerequisites: module 9
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# Семінар Б: Метрики в Умовах Екстремального Дисбалансу Класів

Accuracy = 99.9% звучить імпозантно. Але що, якщо це досягається завдяки тому, що модель просто класифікує все як "Normal" на датасеті з 0.1% критичних логів?

Цей семінар розкриває математичні та практичні аспекти оцінки моделей на екстремально незбалансованих даних, де Base Rate (частота позитивного класу) менше 0.1%.

## Чому Це Важливо?

### Проблема Екстремального Дисбалансу

**Реальний сценарій:**
- Датасет: 1,000,000 логів
- Критичні збої: 1,000 (0.1%)
- Нормальні логи: 999,000 (99.9%)

**Наївна модель:** Класифікує все як "Normal"
- **Accuracy:** 99.9% (оманливо висока!)
- **Recall:** 0% (не знайдено жодного критичного збою)
- **Precision:** undefined (0/0)

**Висновок:** Accuracy оманлива на екстремально незбалансованих даних.

### Base Rate Fallacy в Дії: Рекап через Пастку Байєса

**Важливо:** Цей семінар спирається на математичний фундамент, розглянутий у [Розділі 00: Пастка Байєса](./00_the_bayesian_trap.md). Рекомендується ознайомитися з цим розділом для повного розуміння.

**Ключова ідея Пастки Байєса:** Навіть при високій точності тесту (99%), якщо подія рідкісна (низький Base Rate), більшість позитивних результатів будуть помилковими тривогами.

**Приклад з медицини (з [Розділу 00](./00_the_bayesian_trap.md)):**
- Хвороба рідкісна: $P(\text{Disease}) = 0.001$ (0.1%)
- Тест точний: $P(\text{Positive}|\text{Disease}) = 0.99$, $P(\text{Negative}|\text{Healthy}) = 0.99$
- **Результат:** $P(\text{Disease}|\text{Positive}) \approx 0.09$ (лише 9% справжніх хворих серед позитивних результатів!)

**Аналогія з логами:**
- Критичний збій рідкісний: $P(\text{Critical}) = 0.001$ (0.1%)
- Модель точна: $P(\text{Predicted Critical}|\text{Critical}) = 0.99$ (Recall = 99%)
- Модель точна: $P(\text{Predicted Normal}|\text{Normal}) = 0.99$ (Specificity = 99%)
- **Результат:** $P(\text{Critical}|\text{Predicted Critical}) \approx 0.09$ (лише 9% справжніх критичних серед передбачених!)

**Висновок:** Base Rate Fallacy пояснює, чому навіть дуже точні моделі дають багато помилкових тривог на екстремально незбалансованих даних. Це не помилка моделі — це математика.

## Математика: Base Rate та Precision

### Виведення через Теорему Байєса

**Теорема Байєса:**

$$P(\text{Critical}|\text{Predicted Critical}) = \frac{P(\text{Predicted Critical}|\text{Critical}) \cdot P(\text{Critical})}{P(\text{Predicted Critical})}$$

**Розширення знаменника через формулу повної ймовірності:**

$$P(\text{Predicted Critical}) = P(\text{Predicted Critical}|\text{Critical}) \cdot P(\text{Critical}) + P(\text{Predicted Critical}|\text{Normal}) \cdot P(\text{Normal})$$

**Підставляємо:**

$$P(\text{Critical}|\text{Predicted Critical}) = \frac{P(\text{Predicted Critical}|\text{Critical}) \cdot P(\text{Critical})}{P(\text{Predicted Critical}|\text{Critical}) \cdot P(\text{Critical}) + P(\text{Predicted Critical}|\text{Normal}) \cdot P(\text{Normal})}$$

**Введемо позначення:**
- $P(\text{Critical}) = \pi$ (Base Rate)
- $P(\text{Predicted Critical}|\text{Critical}) = \text{Recall} = R$ (True Positive Rate)
- $P(\text{Predicted Critical}|\text{Normal}) = \text{FPR} = F$ (False Positive Rate)
- $P(\text{Critical}|\text{Predicted Critical}) = \text{Precision} = P$

**Отримуємо:**

$$P = \frac{R \cdot \pi}{R \cdot \pi + F \cdot (1 - \pi)}$$

**Альтернативна форма (ділення чисельника та знаменника на $\pi$):**

$$P = \frac{R}{R + F \cdot \frac{1 - \pi}{\pi}} = \frac{R}{R + F \cdot \frac{1}{\pi} - F}$$

**Для екстремально низького Base Rate ($\pi \ll 1$):**

$$\frac{1 - \pi}{\pi} \approx \frac{1}{\pi}$$

**Отже:**

$$P \approx \frac{R}{R + F \cdot \frac{1}{\pi}}$$

### Аналіз Формули

**Висновок 1:** Precision залежить від Base Rate $\pi$.

**Висновок 2:** При $\pi \to 0$ (екстремальний дисбаланс):

$$P \approx \frac{R}{R + F \cdot \frac{1}{\pi}} \to 0$$

**Це означає:** Навіть при високому Recall та низькому FPR, якщо Base Rate дуже низький, Precision буде низькою.

**Приклад числовий:**

- $\pi = 0.001$ (0.1%)
- $R = 0.95$ (Recall = 95%)
- $F = 0.01$ (FPR = 1%)

$$P = \frac{0.95 \cdot 0.001}{0.95 \cdot 0.001 + 0.01 \cdot 0.999} = \frac{0.00095}{0.00095 + 0.00999} = \frac{0.00095}{0.01094} \approx 0.087$$

**Результат:** Precision ≈ 8.7% (91.3% помилкових тривог!)

**Висновок:** Навіть при дуже хорошому Recall (95%) та низькому FPR (1%), при Base Rate 0.1% більшість передбачень критичних будуть помилковими.

### Граничний Випадок: Ідеальна Модель

**Якщо $F = 0$ (немає False Positives):**

$$P = \frac{R \cdot \pi}{R \cdot \pi + 0} = 1$$

**Висновок:** Тільки ідеальна модель (без False Positives) може мати високу Precision на екстремально незбалансованих даних.

**Реальність:** Навіть найкращі моделі мають $F > 0$, тому Precision завжди буде низькою при дуже низькому Base Rate.

## ROC-AUC vs Precision-Recall Curves

### ROC Curve (Receiver Operating Characteristic)

**Визначення:**
- **X-вісь:** False Positive Rate (FPR) = $\frac{FP}{FP + TN}$
- **Y-вісь:** True Positive Rate (TPR) = Recall = $\frac{TP}{TP + FN}$

**ROC-AUC:** Площа під ROC кривою.

**Властивості:**
- Діапазон: [0, 1]
- ROC-AUC = 0.5: Випадкова модель
- ROC-AUC = 1.0: Ідеальна модель

### Precision-Recall Curve (PRC)

**Визначення:**
- **X-вісь:** Recall = $\frac{TP}{TP + FN}$
- **Y-вісь:** Precision = $\frac{TP}{TP + FP}$

**PR-AUC:** Площа під Precision-Recall кривою.

**Властивості:**
- Діапазон: [0, 1]
- PR-AUC залежить від Base Rate
- PR-AUC = Base Rate: Випадкова модель

### Чому PRC Більш Інформативна для Рідкісних Подій?

#### Проблема з ROC-AUC на Незбалансованих Даних

**Приклад:** Датасет з 0.1% критичних логів.

**ROC-AUC може бути високим навіть для поганої моделі:**

```
Модель A: ROC-AUC = 0.95
  - TPR = 0.90 (знаходить 90% критичних)
  - FPR = 0.10 (10% нормальних класифіковано як критичні)
  
  При Base Rate = 0.001:
  - Precision = 0.0009 / (0.0009 + 0.0999) ≈ 0.009 (0.9%!)
  
  Висновок: ROC-AUC високий, але Precision критично низька!
```

**Проблема:** ROC-AUC не враховує Base Rate. Він фокусується на відносній здатності розрізняти класи, але не на абсолютній якості передбачень.

#### Переваги PRC для Рідкісних Подій

**1. PRC враховує Base Rate:**

PR-AUC завжди нижче для незбалансованих даних, що відображає реальну складність задачі.

**2. PRC фокусується на позитивному класі:**

Для рідкісних подій (критичні збої) важливо саме те, наскільки точно ми їх виявляємо, а не загальна здатність розрізняти класи.

**3. PRC чутлива до змін на незбалансованих даних:**

Невеликі зміни в моделі призводять до помітних змін у PRC, тоді як ROC-AUC може залишатися стабільним.

### Математичне Порівняння

**ROC-AUC:** Інтеграл від FPR до TPR

$$\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}$$

**PR-AUC:** Інтеграл від Recall до Precision

$$\text{PR-AUC} = \int_0^1 \text{Precision}(\text{Recall}) \, d\text{Recall}$$

**Зв'язок між ними:**

Для незбалансованих даних ($\pi \ll 1$):

$$\text{PR-AUC} \approx \pi \cdot \text{ROC-AUC}$$

**Висновок:** PR-AUC пропорційний Base Rate, що робить його більш реалістичним для незбалансованих даних.

## Практична Реалізація

### Генерація Синтетичних Даних з Різними Base Rates

```python
"""
Демонстрація метрик на екстремально незбалансованих даних.
Порівняння ROC-AUC та PR-AUC для різних Base Rates.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
```

### Функція для Обчислення Метрик

```python
def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, base_rate: float) -> dict:
    """
    Обчислює ROC-AUC та PR-AUC для даних.
    
    Args:
        y_true: Істинні мітки
        y_scores: Ймовірності позитивного класу
        base_rate: Base Rate (частота позитивного класу)
    
    Returns:
        Словник з метриками
    """
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    # Baseline для PR-AUC (випадкова модель)
    pr_baseline = base_rate
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'pr_auc': pr_auc,
        'pr_baseline': pr_baseline,
        'roc_thresholds': roc_thresholds,
        'pr_thresholds': pr_thresholds
    }
```

### Візуалізація ROC та PR Curves

```python
def plot_roc_pr_curves(
    metrics_list: list,
    base_rates: list,
    labels: list = None
) -> None:
    """
    Візуалізує ROC та PR curves для різних Base Rates.
    
    Args:
        metrics_list: Список словників з метриками
        base_rates: Список Base Rates
        labels: Мітки для легенди
    """
    if labels is None:
        labels = [f"Base Rate = {br:.4f}" for br in base_rates]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curves
    ax1 = axes[0]
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax1.plot(
            metrics['fpr'],
            metrics['tpr'],
            label=f"{label}\nROC-AUC = {metrics['roc_auc']:.4f}",
            linewidth=2
        )
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=1)
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
    ax1.set_title('ROC Curves for Different Base Rates', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Precision-Recall Curves
    ax2 = axes[1]
    for i, (metrics, label, br) in enumerate(zip(metrics_list, labels, base_rates)):
        ax2.plot(
            metrics['recall'],
            metrics['precision'],
            label=f"{label}\nPR-AUC = {metrics['pr_auc']:.4f}",
            linewidth=2
        )
        # Baseline (випадкова модель)
        ax2.axhline(
            y=br,
            color=f'C{i}',
            linestyle='--',
            alpha=0.5,
            label=f"Baseline = {br:.4f}"
        )
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves for Different Base Rates', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('roc_pr_comparison.png', dpi=300, bbox_inches='tight')
    print("Графік збережено: roc_pr_comparison.png")
    plt.show()
```

### Демонстрація на Різних Base Rates

```python
def demonstrate_base_rate_effect():
    """
    Демонструє вплив Base Rate на ROC-AUC та PR-AUC.
    """
    print("=" * 80)
    print("ДЕМОНСТРАЦІЯ: ВПЛИВ BASE RATE НА МЕТРИКИ")
    print("=" * 80)
    print()
    
    base_rates = [0.5, 0.1, 0.01, 0.001]  # 50%, 10%, 1%, 0.1%
    metrics_list = []
    
    for base_rate in base_rates:
        print(f"Генерація датасету з Base Rate = {base_rate:.4f} ({base_rate*100:.2f}%)...")
        
        # Генеруємо синтетичні дані
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            n_informative=10,
            n_redundant=10,
            n_clusters_per_class=1,
            weights=[1 - base_rate, base_rate],
            random_state=42,
            flip_y=0.01  # Шум
        )
        
        # Розділяємо на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Навчаємо модель
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Передбачаємо ймовірності
        y_scores = model.predict_proba(X_test)[:, 1]
        
        # Обчислюємо метрики
        metrics = compute_metrics(y_test, y_scores, base_rate)
        metrics_list.append(metrics)
        
        # Виводимо статистику
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
        print(f"  Baseline (випадкова модель): {metrics['pr_baseline']:.4f}")
        print(f"  PR-AUC / Baseline: {metrics['pr_auc'] / metrics['pr_baseline']:.2f}x")
        print()
    
    # Візуалізуємо
    print("Створення графіків...")
    plot_roc_pr_curves(metrics_list, base_rates)
    
    # Порівняльна таблиця
    print("=" * 80)
    print("ПОРІВНЯЛЬНА ТАБЛИЦЯ")
    print("=" * 80)
    print(f"{'Base Rate':<15} {'ROC-AUC':<15} {'PR-AUC':<15} {'Baseline':<15} {'PR-AUC/Baseline':<20}")
    print("-" * 80)
    for br, m in zip(base_rates, metrics_list):
        print(f"{br:<15.4f} {m['roc_auc']:<15.4f} {m['pr_auc']:<15.4f} {m['pr_baseline']:<15.4f} {m['pr_auc']/m['pr_baseline']:<20.2f}")
    print()
    
    print("=" * 80)
    print("ВИСНОВКИ:")
    print("=" * 80)
    print("1. ROC-AUC залишається високим навіть при низькому Base Rate")
    print("2. PR-AUC зменшується пропорційно Base Rate")
    print("3. PR-AUC більш реалістично відображає складність задачі")
    print("4. Для екстремально незбалансованих даних (Base Rate < 0.1%) PR-AUC критично важливіша")
    print("=" * 80)
```

### Візуалізація Залежності Precision від Base Rate

```python
def plot_precision_vs_base_rate():
    """
    Візуалізує залежність Precision від Base Rate за формулою Байєса.
    """
    print("=" * 80)
    print("ВІЗУАЛІЗАЦІЯ: PRECISION vs BASE RATE (ФОРМУЛА БАЙЄСА)")
    print("=" * 80)
    print()
    
    # Параметри моделі
    recall_values = [0.8, 0.9, 0.95, 0.99]  # Різні значення Recall
    fpr_values = [0.01, 0.05, 0.1]  # Різні значення FPR
    
    # Base Rates від 0.0001 до 0.5
    base_rates = np.logspace(-4, -0.3, 100)  # Від 0.0001 до 0.5
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Графік 1: Precision vs Base Rate для різних Recall (FPR = 0.01)
    ax1 = axes[0]
    fpr_fixed = 0.01
    for recall in recall_values:
        precision = (recall * base_rates) / (recall * base_rates + fpr_fixed * (1 - base_rates))
        ax1.plot(base_rates, precision, label=f'Recall = {recall:.2f}', linewidth=2)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Base Rate (π)', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'Precision vs Base Rate (FPR = {fpr_fixed:.2f})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Графік 2: Precision vs Base Rate для різних FPR (Recall = 0.95)
    ax2 = axes[1]
    recall_fixed = 0.95
    for fpr in fpr_values:
        precision = (recall_fixed * base_rates) / (recall_fixed * base_rates + fpr * (1 - base_rates))
        ax2.plot(base_rates, precision, label=f'FPR = {fpr:.2f}', linewidth=2)
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Base Rate (π)', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title(f'Precision vs Base Rate (Recall = {recall_fixed:.2f})', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('precision_vs_base_rate.png', dpi=300, bbox_inches='tight')
    print("Графік збережено: precision_vs_base_rate.png")
    plt.show()
    
    # Демонстрація для конкретного Base Rate
    print("\nДЕМОНСТРАЦІЯ ДЛЯ BASE RATE = 0.001 (0.1%):")
    print("-" * 80)
    br_demo = 0.001
    recall_demo = 0.95
    fpr_demo = 0.01
    
    precision_demo = (recall_demo * br_demo) / (recall_demo * br_demo + fpr_demo * (1 - br_demo))
    
    print(f"Base Rate:     {br_demo:.4f} ({br_demo*100:.2f}%)")
    print(f"Recall:        {recall_demo:.2f} ({recall_demo*100:.0f}%)")
    print(f"FPR:           {fpr_demo:.2f} ({fpr_demo*100:.0f}%)")
    print(f"Precision:     {precision_demo:.4f} ({precision_demo*100:.2f}%)")
    print(f"False Alarms:  {(1 - precision_demo)*100:.2f}%")
    print()
    print("Висновок: Навіть при високому Recall (95%) та низькому FPR (1%),")
    print(f"          Precision становить лише {precision_demo*100:.2f}% при Base Rate 0.1%.")
```

### Головна Функція

```python
def main():
    """
    Головна функція для демонстрації метрик на екстремально незбалансованих даних.
    """
    print("=" * 80)
    print("СЕМІНАР Б: МЕТРИКИ В УМОВАХ ЕКСТРЕМАЛЬНОГО ДИСБАЛАНСУ КЛАСІВ")
    print("=" * 80)
    print()
    
    # Демонстрація впливу Base Rate
    demonstrate_base_rate_effect()
    
    print("\n" + "=" * 80 + "\n")
    
    # Візуалізація залежності Precision від Base Rate
    plot_precision_vs_base_rate()
    
    print("\n" + "=" * 80)
    print("КЛЮЧОВІ ВИСНОВКИ:")
    print("=" * 80)
    print("1. Accuracy оманлива на екстремально незбалансованих даних")
    print("2. Precision залежить від Base Rate через формулу Байєса")
    print("3. PR-AUC більш інформативна за ROC-AUC для рідкісних подій")
    print("4. При Base Rate < 0.1% навіть хороші моделі мають низьку Precision")
    print("5. PR Curves краще відображають реальну якість моделі на незбалансованих даних")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

## Порівняння ROC-AUC та PR-AUC: Практичний Приклад

### Сценарій: Детекція Критичних Збоїв

**Датасет:**
- 1,000,000 логів
- 1,000 критичних (0.1%)
- 999,000 нормальних (99.9%)

**Модель A:**
- ROC-AUC = 0.95
- PR-AUC = 0.15
- При threshold = 0.5:
  - Precision = 0.08 (8%)
  - Recall = 0.90 (90%)

**Модель B:**
- ROC-AUC = 0.92
- PR-AUC = 0.25
- При threshold = 0.5:
  - Precision = 0.12 (12%)
  - Recall = 0.85 (85%)

**Питання:** Яка модель краща?

**Відповідь:** Модель B краща для цієї задачі, хоча має нижчий ROC-AUC.

**Чому?**
- PR-AUC моделі B вища (0.25 vs 0.15)
- Precision моделі B вища (12% vs 8%)
- Для Alert Fatigue важливіше мати менше помилкових тривог (вища Precision)

**Висновок:** ROC-AUC може вводити в оману на незбалансованих даних. PR-AUC дає більш реалістичну оцінку.

## Matthews Correlation Coefficient (MCC): Математичне Виведення Переваг над ROC-AUC

### Визначення MCC

**Matthews Correlation Coefficient:**

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

**Властивості MCC:**
- Діапазон: $[-1, +1]$
- $MCC = +1$: Ідеальна класифікація
- $MCC = 0$: Випадкова класифікація
- $MCC = -1$: Повна протилежність

### Чому MCC Краще за ROC-AUC для Незбалансованих Даних

#### 1. MCC Враховує Всі Чотири Клітинки Confusion Matrix

**ROC-AUC:** Залежить лише від TPR (Recall) та FPR:
- TPR = $\frac{TP}{TP + FN}$
- FPR = $\frac{FP}{FP + TN}$

**Проблема:** ROC-AUC не враховує абсолютні значення TP, TN, FP, FN, лише їх відносні співвідношення.

**MCC:** Враховує всі чотири значення:
- Чисельник: $TP \times TN - FP \times FN$ (враховує всі чотири)
- Знаменник: $\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}$ (нормалізація)

**Висновок:** MCC дає повну картину класифікації, а не лише відносні співвідношення.

#### 2. MCC Інваріантний до Незбалансованості

**Приклад:** Датасет з 0.1% критичних логів.

**Сценарій 1:** Модель класифікує все як Normal
- TP = 0, TN = 999,000, FP = 0, FN = 1,000
- **ROC-AUC:** Невизначений (TPR = 0, FPR = 0 → точка на початку кривої)
- **MCC:** $MCC = \frac{0 \times 999000 - 0 \times 1000}{\sqrt{(0+0)(0+1000)(999000+0)(999000+1000)}} = 0$ (випадкова класифікація)

**Сценарій 2:** Модель знаходить 50% критичних без помилок
- TP = 500, TN = 999,000, FP = 0, FN = 500
- **ROC-AUC:** Високий (TPR = 0.5, FPR = 0)
- **MCC:** $MCC = \frac{500 \times 999000 - 0 \times 500}{\sqrt{(500+0)(500+500)(999000+0)(999000+500)}} \approx 0.71$ (хороша якість)

**Висновок:** MCC дає змістовний результат навіть коли ROC-AUC невизначений.

#### 3. Математичне Порівняння: MCC vs ROC-AUC

**ROC-AUC:** Інтеграл від FPR до TPR

$$\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d\text{FPR}$$

**Проблема:** ROC-AUC не залежить від Base Rate $\pi$. Для незбалансованих даних це може бути оманливим.

**MCC:** Безпосередньо залежить від структури confusion matrix, яка залежить від Base Rate.

**Формалізація:**

Для датасету з $N$ зразками та Base Rate $\pi$:
- Очікувана кількість позитивних: $N \times \pi$
- Очікувана кількість негативних: $N \times (1 - \pi)$

**ROC-AUC:** Не залежить від $\pi$ (лише від відносних співвідношень)

**MCC:** Залежить від $\pi$ через структуру confusion matrix:
- При $\pi \to 0$: TN домінує → MCC більш чутливий до помилок у TP/FN
- При $\pi \to 1$: TP домінує → MCC більш чутливий до помилок у TN/FP

**Висновок:** MCC автоматично враховує незбалансованість через структуру confusion matrix.

#### 4. Практичний Приклад: MCC vs ROC-AUC

**Датасет:** 1,000,000 логів, 1,000 критичних (0.1%)

**Модель A:**
- TP = 900, TN = 999,000, FP = 0, FN = 100
- **ROC-AUC:** 0.999 (дуже високий!)
- **MCC:** $\frac{900 \times 999000 - 0 \times 100}{\sqrt{(900+0)(900+100)(999000+0)(999000+100)}} \approx 0.95$

**Модель B:**
- TP = 900, TN = 980,000, FP = 19,000, FN = 100
- **ROC-AUC:** 0.95 (нижчий за A)
- **MCC:** $\frac{900 \times 980000 - 19000 \times 100}{\sqrt{(900+19000)(900+100)(980000+19000)(980000+100)}} \approx 0.43$

**Аналіз:**
- Модель A має вищий ROC-AUC та вищий MCC → краща модель
- Але якщо порівняти лише ROC-AUC, різниця невелика (0.999 vs 0.95)
- MCC показує значно більшу різницю (0.95 vs 0.43) → більш чутлива метрика

**Висновок:** MCC краще розрізняє якість моделей на незбалансованих даних.

### Порівняльна Таблиця: MCC vs ROC-AUC vs PR-AUC

| Метрика | Діапазон | Враховує Base Rate? | Враховує всі 4 клітинки? | Чутлива до незбалансованості? |
|---------|----------|---------------------|--------------------------|-------------------------------|
| ROC-AUC | [0, 1] | Ні | Ні (лише TPR, FPR) | Ні (оманлива) |
| PR-AUC | [0, 1] | Так (через Precision) | Ні (лише TP, FP, FN) | Так |
| **MCC** | **[-1, 1]** | **Так (через структуру CM)** | **Так (TP, TN, FP, FN)** | **Так** |

**Висновок:** MCC — найповніша метрика для незбалансованих даних, оскільки:
1. Враховує всі чотири клітинки confusion matrix
2. Автоматично враховує Base Rate через структуру confusion matrix
3. Дає змістовний результат навіть при екстремальній незбалансованості
4. Симетрична та інваріантна до вибору класу

**Рекомендація:** Для екстремально незбалансованих даних (Base Rate < 0.1%) використовуйте MCC як основну метрику, доповнюючи її PR-AUC для аналізу Precision-Recall trade-off.

## Ключові Висновки

1. **Accuracy оманлива:** На екстремально незбалансованих даних висока Accuracy не означає хорошу модель.

2. **Precision залежить від Base Rate:** Через формулу Байєса, навіть при високому Recall та низькому FPR, Precision буде низькою при дуже низькому Base Rate.

3. **PR-AUC більш інформативна:** Для рідкісних подій (Base Rate < 0.1%) Precision-Recall Curves дають більш реалістичну оцінку якості моделі, ніж ROC Curves.

4. **Математичний зв'язок:** $P = \frac{R \cdot \pi}{R \cdot \pi + F \cdot (1 - \pi)}$, де $\pi$ — Base Rate, $R$ — Recall, $F$ — FPR.

5. **Практичне застосування:** Для Alert Fatigue важливіше максимізувати PR-AUC, а не ROC-AUC, оскільки це мінімізує кількість помилкових тривог.

## Рекомендована Література

### Precision-Recall Curves vs ROC Curves

1. **Saito, T., & Rehmsmeier, M.** (2015). "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets"
   - PLOS ONE, 10(3), e0118432.
   - **Ключова робота:** Детальне порівняння PRC та ROC для незбалансованих даних.

2. **Davis, J., & Goadrich, M.** (2006). "The relationship between Precision-Recall and ROC curves"
   - ICML. Математичний зв'язок між PRC та ROC.

### Base Rate та Теорема Байєса

3. **Provost, F., & Fawcett, T.** (2013). "Data Science for Business"
   - O'Reilly Media. Розділ 4: "Evaluating Predictive Models" — детальний розбір метрик.

4. **Fawcett, T.** (2006). "An introduction to ROC analysis"
   - Pattern Recognition Letters, 27(8), 861-874.
   - Класична робота про ROC analysis.

### Практичні Застосування

5. **He, H., & Garcia, E. A.** (2009). "Learning from imbalanced data"
   - IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263-1284.
   - Огляд методів роботи з незбалансованими даними.

6. **Branco, P., Torgo, L., & Ribeiro, R. P.** (2016). "A survey of predictive modeling on imbalanced domains"
   - ACM Computing Surveys, 49(2), 1-50.
   - Сучасний огляд методів та метрик.

### Математична Теорія

7. **Hand, D. J., & Till, R. J.** (2001). "A simple generalisation of the area under the ROC curve for multiple class classification problems"
   - Machine Learning, 45(2), 171-186.
   - Розширення ROC-AUC для багатокласових задач.

8. **Flach, P., & Kull, M.** (2015). "Precision-recall-gain curves: PR analysis done right"
   - NIPS. Покращена версія PR curves.

---

**Примітка для студентів:** Почніть з Saito & Rehmsmeier (2015) для розуміння, чому PRC краща за ROC на незбалансованих даних. Для математичного виведення формули Precision через Base Rate використовуйте Davis & Goadrich (2006). Для практичних застосувань дивіться Provost & Fawcett (2013) та He & Garcia (2009).

