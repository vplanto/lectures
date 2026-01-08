---
title: "Пастка Байєса: Чому 99% Точність = 90% Помилок"
layout: default
author: Віталій Платонов
---

# Пастка Байєса: Чому 99% Точність = 90% Помилок

Уявіть: медичний тест на рідкісну хворобу має точність 99%. Ви отримуєте позитивний результат. Яка ймовірність, що ви дійсно хворі?

Інтуїція каже: 99%. Математика каже: близько 9%.

Це не помилка в розрахунках. Це фундаментальна пастка, яка руйнує системи моніторингу, спам-фільтри та діагностичні алгоритми. Розберемося, чому.

## Парадокс Медичного Тесту

**Умова задачі:**

- Хвороба трапляється у 1 з 1000 людей (base rate = 0.1%)
- Тест має точність 99%:
  - Якщо людина хвора → тест позитивний у 99% випадків (True Positive Rate = 0.99)
  - Якщо людина здорова → тест негативний у 99% випадків (True Negative Rate = 0.99)

**Питання:** Якщо тест показав позитивний результат, яка ймовірність, що людина дійсно хвора?

### Інтуїтивна (і помилкова) відповідь

Більшість людей відповідають: "99%". Логіка здається очевидною: тест точний на 99%, отже, якщо він показав позитивний результат, то з ймовірністю 99% людина хвора.

### Правильна відповідь через теорему Байєса

Позначимо:
- $P(D) = 0.001$ — ймовірність хвороби (prior probability)
- $P(\neg D) = 0.999$ — ймовірність здоров'я
- $P(+|D) = 0.99$ — ймовірність позитивного тесту, якщо хворий (sensitivity)
- $P(-|\neg D) = 0.99$ — ймовірність негативного тесту, якщо здоровий (specificity)
- $P(+|\neg D) = 0.01$ — ймовірність помилкового позитивного результату (False Positive Rate)

Нас цікавить $P(D|+)$ — ймовірність хвороби за умови позитивного тесту.

**Теорема Байєса:**

$$P(D|+) = \frac{P(+|D) \cdot P(D)}{P(+)}$$

**Формула повної ймовірності для $P(+)$:**

$$P(+) = P(+|D) \cdot P(D) + P(+|\neg D) \cdot P(\neg D)$$

Підставляємо значення:

$$P(+) = 0.99 \cdot 0.001 + 0.01 \cdot 0.999 = 0.00099 + 0.00999 = 0.01098$$

Тепер обчислюємо $P(D|+)$:

$$P(D|+) = \frac{0.99 \cdot 0.001}{0.01098} = \frac{0.00099}{0.01098} \approx 0.0902$$

**Відповідь: близько 9%**, а не 99%.

### Чому це відбувається?

Ключова проблема — **base rate fallacy** (пастка базової частоти). Хвороба настільки рідкісна (0.1%), що навіть при високій точності тесту кількість помилкових позитивних результатів переважає кількість справжніх позитивних.

**Чисельний приклад на популяції 100,000 людей:**

- Хворих: $100,000 \times 0.001 = 100$
- Здорових: $100,000 \times 0.999 = 99,900$

**Результати тестування:**

- Справжні позитивні (True Positives): $100 \times 0.99 = 99$
- Помилкові позитивні (False Positives): $99,900 \times 0.01 = 999$

**Загальна кількість позитивних результатів:** $99 + 999 = 1,098$

**Ймовірність хвороби за умови позитивного тесту:**

$$P(D|+) = \frac{99}{1,098} \approx 0.0902 = 9.02\%$$

З 1,098 позитивних результатів лише 99 справжні. Решта 999 — помилкові тривоги.

## False Positive vs False Negative

Важливо розрізняти два типи помилок:

### False Positive (Помилкова тривога)

- Тест показав позитивний результат, але хвороби немає
- У нашому прикладі: 999 помилкових тривог з 1,098 позитивних результатів
- **Наслідок:** Зайва тривога, стрес, непотрібні додаткові обстеження

### False Negative (Пропущена хвороба)

- Тест показав негативний результат, але хвороба є
- У нашому прикладі: $100 \times 0.01 = 1$ пропущена хвороба
- **Наслідок:** Хвороба не виявлена, лікування не розпочато

**Торгівля (trade-off):** Зменшення False Positives часто збільшує False Negatives, і навпаки. Вибір залежить від контексту:
- **Медицина:** Краще помилитися на стороні обережності (знизити False Negatives)
- **DevOps:** Занадто багато помилкових тривог → Alert Fatigue → справжні проблеми ігноруються

## Симуляція Парадоксу

Наступний код демонструє парадокс на великій вибірці:

```python
"""
Симуляція медичного тесту з високою точністю на рідкісну хворобу.
Демонструє Base Rate Fallacy (Пастку Базової Частоти).
"""

from typing import Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class TestResults:
    """Результати тестування з деталізацією помилок."""
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    @property
    def total_positive(self) -> int:
        """Загальна кількість позитивних результатів."""
        return self.true_positives + self.false_positives
    
    @property
    def total_negative(self) -> int:
        """Загальна кількість негативних результатів."""
        return self.true_negatives + self.false_negatives
    
    @property
    def precision(self) -> float:
        """Точність: P(D|+) = TP / (TP + FP)."""
        if self.total_positive == 0:
            return 0.0
        return self.true_positives / self.total_positive
    
    @property
    def recall(self) -> float:
        """Повнота: P(+|D) = TP / (TP + FN)."""
        total_diseased = self.true_positives + self.false_negatives
        if total_diseased == 0:
            return 0.0
        return self.true_positives / total_diseased
    
    @property
    def accuracy(self) -> float:
        """Загальна точність: (TP + TN) / Total."""
        total = (
            self.true_positives + self.false_positives +
            self.true_negatives + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total


def simulate_medical_test(
    population_size: int,
    disease_rate: float,
    test_sensitivity: float,
    test_specificity: float,
    random_seed: int = 42
) -> TestResults:
    """
    Симулює медичний тест на популяції.
    
    Args:
        population_size: Розмір популяції
        disease_rate: Частота хвороби (0.0 - 1.0)
        test_sensitivity: Чутливість тесту P(+|D) (True Positive Rate)
        test_specificity: Специфічність тесту P(-|¬D) (True Negative Rate)
        random_seed: Seed для відтворюваності
    
    Returns:
        TestResults з деталізацією помилок
    """
    np.random.seed(random_seed)
    
    # Генеруємо стан здоров'я (1 = хворий, 0 = здоровий)
    has_disease = np.random.binomial(1, disease_rate, population_size)
    
    # Генеруємо результати тесту
    test_results = np.zeros(population_size, dtype=int)
    
    # Для хворих: тест позитивний з ймовірністю sensitivity
    diseased_mask = has_disease == 1
    test_results[diseased_mask] = np.random.binomial(
        1, test_sensitivity, size=diseased_mask.sum()
    )
    
    # Для здорових: тест негативний з ймовірністю specificity
    # (тобто позитивний з ймовірністю 1 - specificity)
    healthy_mask = has_disease == 0
    test_results[healthy_mask] = np.random.binomial(
        1, 1 - test_specificity, size=healthy_mask.sum()
    )
    
    # Підрахунок результатів
    true_positives = np.sum((has_disease == 1) & (test_results == 1))
    false_positives = np.sum((has_disease == 0) & (test_results == 1))
    true_negatives = np.sum((has_disease == 0) & (test_results == 0))
    false_negatives = np.sum((has_disease == 1) & (test_results == 0))
    
    return TestResults(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives
    )


def demonstrate_paradox() -> None:
    """Демонструє парадокс на класичному прикладі."""
    print("=" * 70)
    print("ПАРАДОКС МЕДИЧНОГО ТЕСТУ: 99% ТОЧНІСТЬ → 9% ВІРНОСТІ")
    print("=" * 70)
    print()
    
    # Параметри з прикладу
    population_size = 100_000
    disease_rate = 0.001  # 1 на 1000
    test_sensitivity = 0.99  # P(+|D)
    test_specificity = 0.99  # P(-|¬D)
    
    print(f"Параметри симуляції:")
    print(f"  Розмір популяції: {population_size:,}")
    print(f"  Частота хвороби: {disease_rate * 100:.1f}%")
    print(f"  Чутливість тесту: {test_sensitivity * 100:.1f}%")
    print(f"  Специфічність тесту: {test_specificity * 100:.1f}%")
    print()
    
    results = simulate_medical_test(
        population_size=population_size,
        disease_rate=disease_rate,
        test_sensitivity=test_sensitivity,
        test_specificity=test_specificity
    )
    
    print("Результати симуляції:")
    print(f"  Хворих у популяції: {results.true_positives + results.false_negatives:,}")
    print(f"  Здорових у популяції: {results.true_negatives + results.false_positives:,}")
    print()
    
    print("Результати тестування:")
    print(f"  True Positives (справжні хворі): {results.true_positives:,}")
    print(f"  False Positives (помилкові тривоги): {results.false_positives:,}")
    print(f"  True Negatives (справжні здорові): {results.true_negatives:,}")
    print(f"  False Negatives (пропущені хворі): {results.false_negatives:,}")
    print()
    
    print("Метрики:")
    print(f"  Загальна точність (Accuracy): {results.accuracy * 100:.2f}%")
    print(f"  Точність позитивних (Precision): {results.precision * 100:.2f}%")
    print(f"  Повнота (Recall): {results.recall * 100:.2f}%")
    print()
    
    print("=" * 70)
    print("ВИСНОВОК:")
    print(f"  Хоча тест має точність {results.accuracy * 100:.1f}%,")
    print(f"  якщо він показав позитивний результат,")
    print(f"  ймовірність дійсної хвороби лише {results.precision * 100:.1f}%!")
    print("=" * 70)


def explore_parameter_space() -> None:
    """Досліджує, як зміна частоти хвороби впливає на результат."""
    print("\n" + "=" * 70)
    print("ВПЛИВ ЧАСТОТИ ХВОРОБИ НА ТОЧНІСТЬ ПОЗИТИВНИХ РЕЗУЛЬТАТІВ")
    print("=" * 70)
    print()
    
    disease_rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
    test_sensitivity = 0.99
    test_specificity = 0.99
    population_size = 100_000
    
    print(f"{'Частота хвороби':<20} {'Precision':<15} {'False Positives':<20}")
    print("-" * 70)
    
    for rate in disease_rates:
        results = simulate_medical_test(
            population_size=population_size,
            disease_rate=rate,
            test_sensitivity=test_sensitivity,
            test_specificity=test_specificity,
            random_seed=int(rate * 10000)  # Різні seed для різних rate
        )
        print(
            f"{rate * 100:>6.2f}%{'':<12} "
            f"{results.precision * 100:>6.2f}%{'':<8} "
            f"{results.false_positives:>8,}"
        )


if __name__ == "__main__":
    demonstrate_paradox()
    explore_parameter_space()
```

**Очікуваний вивід:**

```
======================================================================
ПАРАДОКС МЕДИЧНОГО ТЕСТУ: 99% ТОЧНІСТЬ → 9% ВІРНОСТІ
======================================================================

Параметри симуляції:
  Розмір популяції: 100,000
  Частота хвороби: 0.1%
  Чутливість тесту: 99.0%
  Специфічність тесту: 99.0%

Результати симуляції:
  Хворих у популяції: ~100
  Здорових у популяції: ~99,900

Результати тестування:
  True Positives (справжні хворі): ~99
  False Positives (помилкові тривоги): ~999
  True Negatives (справжні здорові): ~98,901
  False Negatives (пропущені хворі): ~1

Метрики:
  Загальна точність (Accuracy): 99.00%
  Точність позитивних (Precision): 9.02%
  Повнота (Recall): 99.00%
```

## Зв'язок з DevOps: Alert Fatigue

Тепер перенесемо цю логіку на технічний домен.

**Аналогія:**

- **Хвороба** = Критичний збій системи (наприклад, падіння бази даних)
- **Здоров'я** = Нормальна робота системи
- **Медичний тест** = Система моніторингу, яка сканує логи
- **Позитивний результат** = Alert (сповіщення про помилку)

**Типова ситуація в продакшені:**

- Критичні збої трапляються рідко: 1 на 10,000 подій (0.01%)
- Система моніторингу має високу точність: 99%
- Але через base rate fallacy: з 100 сповіщень лише 1 справжнє

**Наслідок:** Інженери отримують 99 помилкових тривог на кожну справжню проблему. Через кілька тижнів вони починають ігнорувати всі сповіщення — це **Alert Fatigue**.

**Математика:**

$$P(\text{Критичний збій} | \text{Alert}) = \frac{P(\text{Alert} | \text{Збій}) \cdot P(\text{Збій})}{P(\text{Alert})}$$

$$P(\text{Alert}) = 0.99 \cdot 0.0001 + 0.01 \cdot 0.9999 \approx 0.0101$$

$$P(\text{Збій} | \text{Alert}) = \frac{0.99 \cdot 0.0001}{0.0101} \approx 0.0098 = 0.98\%$$

**Висновок:** Навіть при 99% точності моніторингу, якщо критичні збої рідкісні, більшість сповіщень будуть помилковими. Це руйнує довіру до системи.

## Ключові Висновки

1. **Accuracy ≠ Precision:** Висока загальна точність не гарантує високої точності позитивних результатів.

2. **Base Rate має значення:** Рідкісні події вимагають особливої обережності при інтерпретації результатів тестування.

3. **False Positives руйнують системи:** У технічних доменах помилкові тривоги часто гірші за пропущені події, бо призводять до Alert Fatigue.

4. **Потрібні інші метрики:** Accuracy недостатня. Потрібні Precision, Recall, F1-score, особливо на незбалансованих даних.

У наступному розділі ми подивимося, як ця проблема проявляється в реальних системах моніторингу та як її вирішують через машинне навчання.

## Рекомендована Література

### Відео-матеріали

1. **Veritasium. "The Bayesian Trap"** (2021)
   - URL: https://www.youtube.com/watch?v=R13BD8qKeTg
   - Класичне пояснення парадоксу медичного тесту з візуалізацією. Основа для цього розділу.

2. **3Blue1Brown. "Bayes theorem, and making probability intuitive"** (2020)
   - URL: https://www.youtube.com/watch?v=HZGCoVF3YvM
   - Геометрична інтерпретація теореми Байєса через діаграми Венна.

### Класичні Тексти

3. **Bayes, T.** (1763). "An Essay towards solving a Problem in the Doctrine of Chances"
   - Оригінальна робота Томаса Байєса. Історичний контекст.

4. **Kahneman, D.** (2011). "Thinking, Fast and Slow"
   - Розділ про когнітивні упередження та base rate neglect. Пояснює, чому інтуїція підводить при оцінці рідкісних подій.

5. **MacKay, D. J. C.** (2003). "Information Theory, Inference, and Learning Algorithms"
   - Розділ 3: "More about Inference". Строга математична формалізація байєсівського висновування.

### Технічна Література

6. **Provost, F., & Fawcett, T.** (2013). "Data Science for Business"
   - Розділ 4: "Evaluating Predictive Models". Precision, Recall, F1-score на незбалансованих даних.

7. **Sculley, D., et al.** (2015). "Hidden Technical Debt in Machine Learning Systems"
   - NIPS 2015. Обговорює проблему "data dependencies" та чому метрики навчальної вибірки не відображають продакшн-реальність.

8. **Kim, G., et al.** (2016). "The DevOps Handbook"
   - Розділ про моніторинг та Alert Fatigue. Практичні стратегії зменшення помилкових тривог.

### Академічні Статті

9. **Bar-Hillel, M.** (1980). "The base-rate fallacy in probability judgments"
   - Acta Psychologica, 44(3), 211-233. Психологічне дослідження помилок при оцінці базової частоти.

10. **Gigerenzer, G., & Hoffrage, U.** (1995). "How to improve Bayesian reasoning without instruction: Frequency formats"
    - Psychological Review, 102(4), 684-704. Демонструє, що "frequency format" покращує розуміння байєсівських задач.

### Онлайн-Ресурси

11. **Wikipedia. "Base rate fallacy"**
    - URL: https://en.wikipedia.org/wiki/Base_rate_fallacy
    - Швидкий довідник з прикладами з різних доменів.

12. **Scikit-learn Documentation. "Classification metrics"**
    - URL: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    - Практичний гайд з реалізацією метрик у Python.

---

**Примітка для студентів:** Почніть з відео Veritasium для інтуїтивного розуміння, потім перейдіть до MacKay для математичної строгості. Для практичної реалізації використовуйте документацію Scikit-learn.

