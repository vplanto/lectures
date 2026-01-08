---
title: "Stability Evaluation: Оцінка стабільності пояснень (Robustness)"
layout: default
nav_order: 10
---

# Stability Evaluation: Оцінка стабільності пояснень (Robustness)

> **Частина курсу:** [Інтерпретований ШІ (XAI) та Прескриптивний Аналіз](./index.md)  
> **Попередні матеріали:** [Семінар 2: SHAP та LIME](./08_seminar2_shap_lime.md), [Семінар 3: Performance Doctor](./09_seminar3_performance_doctor.md)  
> **Важливість:** Обов'язковий етап валідації системи Performance Doctor

## Методологічний контекст

### Проблема нестабільності пояснень

**Критична проблема:** Пояснення (SHAP, LIME) можуть бути **нестабільними** — різні запуски можуть давати різні результати для того самого спостереження.

**Приклад:**
```python
# Запуск 1: SHAP каже, що cpu_usage додав +200ms
# Запуск 2: SHAP каже, що cpu_usage додав +150ms
# Запуск 3: SHAP каже, що ram_usage додав +180ms

# Якому результату довіряти?
```

**Наслідки нестабільності:**
- ❌ Неможливо прийняти надійні рішення на основі пояснень
- ❌ Рекомендації можуть бути суперечливими
- ❌ Втрата довіри до системи Performance Doctor

### Концепція Robustness (Стабільність)

**Robustness** — це міра того, наскільки стабільні пояснення при невеликих змінах вхідних даних або параметрів алгоритму.

**Властивості стабільного пояснення:**
1. ✅ **Відтворюваність:** Однакові результати при однакових входах
2. ✅ **Стійкість до шуму:** Невеликі зміни даних не змінюють суттєво пояснення
3. ✅ **Узгодженість:** Різні методи (SHAP, LIME) дають схожі результати

---

## Математична формалізація

### Визначення Robustness

Нехай:
- $\mathbf{x}$ — спостереження для пояснення
- $\phi(\mathbf{x})$ — пояснення (наприклад, SHAP значення)
- $\mathcal{N}_\epsilon(\mathbf{x}) = \{\mathbf{x}' : ||\mathbf{x}' - \mathbf{x}|| \leq \epsilon\}$ — $\epsilon$-окіл спостереження

**Robustness** визначається як:

$$\text{Robustness}(\phi, \mathbf{x}, \epsilon) = 1 - \frac{1}{|\mathcal{N}_\epsilon(\mathbf{x})|} \sum_{\mathbf{x}' \in \mathcal{N}_\epsilon(\mathbf{x})} \frac{||\phi(\mathbf{x}') - \phi(\mathbf{x})||}{||\phi(\mathbf{x})|| + \delta}$$

де $\delta > 0$ — мала константа для уникнення ділення на нуль.

**Інтерпретація:**
- $\text{Robustness} \approx 1$ — пояснення стабільне
- $\text{Robustness} \approx 0$ — пояснення нестабільне

### Метрики стабільності

#### 1. Варіативність (Variance)

Варіативність пояснень при багаторазовому запуску:

$$\text{Var}(\phi) = \frac{1}{n-1} \sum_{i=1}^{n} ||\phi_i(\mathbf{x}) - \bar{\phi}(\mathbf{x})||^2$$

де:
- $\phi_i(\mathbf{x})$ — пояснення при $i$-му запуску
- $\bar{\phi}(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^{n} \phi_i(\mathbf{x})$ — середнє пояснення
- $n$ — кількість запусків

#### 2. Коефіцієнт варіації (Coefficient of Variation)

Відносна міра варіативності:

$$\text{CV}(\phi) = \frac{\sqrt{\text{Var}(\phi)}}{||\bar{\phi}(\mathbf{x})|| + \delta}$$

**Критерій стабільності:**
- $\text{CV} < 0.1$ — висока стабільність
- $0.1 \leq \text{CV} < 0.3$ — середня стабільність
- $\text{CV} \geq 0.3$ — низька стабільність

#### 3. Rank Stability (Стабільність рангу)

Стабільність порядку важливості ознак:

$$\text{RankStability}(\phi) = \frac{1}{n(n-1)} \sum_{i=1}^{n} \sum_{j=i+1}^{n} \text{Spearman}(\text{rank}(\phi_i), \text{rank}(\phi_j))$$

де $\text{Spearman}$ — кореляція Спірмена між рангами.

**Інтерпретація:**
- $\text{RankStability} \approx 1$ — порядок важливості стабільний
- $\text{RankStability} \approx 0$ — порядок важливості нестабільний

#### 4. Top-K Stability

Стабільність топ-$k$ найважливіших ознак:

$$\text{TopKStability}(\phi, k) = \frac{1}{n(n-1)} \sum_{i=1}^{n} \sum_{j=i+1}^{n} \frac{|\text{TopK}(\phi_i) \cap \text{TopK}(\phi_j)|}{k}$$

де $\text{TopK}(\phi)$ — множина топ-$k$ ознак за абсолютним значенням внеску.

---

## Методи оцінки стабільності

### Метод 1: Bootstrap Sampling

**Ідея:** Генерувати пояснення на різних bootstrap зразках даних.

```python
def evaluate_robustness_bootstrap(
    explainer,
    instance: pd.DataFrame,
    background_data: pd.DataFrame,
    n_bootstrap: int = 100,
    bootstrap_size: float = 0.8
) -> Dict[str, float]:
    """
    Оцінка стабільності через bootstrap sampling.
    
    Parameters:
    -----------
    explainer : Explainer
        Об'єкт для пояснення (SHAP або LIME)
    instance : pd.DataFrame
        Спостереження для пояснення
    background_data : pd.DataFrame
        Фонові дані
    n_bootstrap : int
        Кількість bootstrap зразків
    bootstrap_size : float
        Розмір bootstrap зразка (частка від background_data)
        
    Returns:
    --------
    metrics : Dict[str, float]
        Метрики стабільності
    """
    explanations = []
    
    for _ in range(n_bootstrap):
        # Генерація bootstrap зразка
        bootstrap_indices = np.random.choice(
            len(background_data),
            size=int(len(background_data) * bootstrap_size),
            replace=True
        )
        bootstrap_data = background_data.iloc[bootstrap_indices]
        
        # Генерація пояснення
        explanation = explainer.explain(instance, bootstrap_data)
        explanations.append(explanation)
    
    # Обчислення метрик
    explanations_array = np.array(explanations)
    mean_explanation = np.mean(explanations_array, axis=0)
    var_explanation = np.var(explanations_array, axis=0)
    
    # Коефіцієнт варіації
    cv = np.sqrt(var_explanation) / (np.abs(mean_explanation) + 1e-10)
    mean_cv = np.mean(cv)
    
    # Rank stability
    ranks = [np.argsort(np.abs(exp))[::-1] for exp in explanations]
    rank_correlations = []
    for i in range(len(ranks)):
        for j in range(i+1, len(ranks)):
            from scipy.stats import spearmanr
            corr, _ = spearmanr(ranks[i], ranks[j])
            rank_correlations.append(corr)
    rank_stability = np.mean(rank_correlations) if rank_correlations else 0.0
    
    return {
        'mean_cv': mean_cv,
        'rank_stability': rank_stability,
        'variance': np.mean(var_explanation)
    }
```

### Метод 2: Perturbation Analysis

**Ідея:** Оцінювати стабільність при невеликих змінах вхідних даних.

```python
def evaluate_robustness_perturbation(
    explainer,
    instance: pd.DataFrame,
    background_data: pd.DataFrame,
    n_perturbations: int = 50,
    noise_level: float = 0.05
) -> Dict[str, float]:
    """
    Оцінка стабільності через perturbation analysis.
    
    Parameters:
    -----------
    explainer : Explainer
        Об'єкт для пояснення
    instance : pd.DataFrame
        Спостереження для пояснення
    background_data : pd.DataFrame
        Фонові дані
    n_perturbations : int
        Кількість perturbations
    noise_level : float
        Рівень шуму (стандартне відхилення відносно середнього)
        
    Returns:
    --------
    metrics : Dict[str, float]
        Метрики стабільності
    """
    base_explanation = explainer.explain(instance, background_data)
    explanations = [base_explanation]
    
    instance_values = instance.values[0]
    std_values = background_data.std().values
    
    for _ in range(n_perturbations):
        # Додавання шуму
        noise = np.random.normal(0, noise_level, size=instance_values.shape)
        perturbed_values = instance_values + noise * std_values
        perturbed_instance = pd.DataFrame([perturbed_values], columns=instance.columns)
        
        # Генерація пояснення
        explanation = explainer.explain(perturbed_instance, background_data)
        explanations.append(explanation)
    
    # Обчислення метрик
    explanations_array = np.array(explanations)
    mean_explanation = np.mean(explanations_array, axis=0)
    
    # Robustness
    deviations = []
    for exp in explanations:
        deviation = np.linalg.norm(exp - mean_explanation) / (np.linalg.norm(mean_explanation) + 1e-10)
        deviations.append(deviation)
    
    robustness = 1.0 - np.mean(deviations)
    
    return {
        'robustness': robustness,
        'mean_deviation': np.mean(deviations),
        'max_deviation': np.max(deviations)
    }
```

### Метод 3: Agreement Between Methods

**Ідея:** Оцінювати узгодженість між різними методами пояснення (SHAP vs LIME).

```python
def evaluate_agreement(
    shap_explainer,
    lime_explainer,
    instance: pd.DataFrame,
    background_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Оцінка узгодженості між SHAP та LIME.
    
    Parameters:
    -----------
    shap_explainer : KernelSHAPExplainer
        SHAP explainer
    lime_explainer : LIMExplainer
        LIME explainer
    instance : pd.DataFrame
        Спостереження для пояснення
    background_data : pd.DataFrame
        Фонові дані
        
    Returns:
    --------
    metrics : Dict[str, float]
        Метрики узгодженості
    """
    # SHAP пояснення
    shap_result = shap_explainer.explain(instance, background_data)
    shap_values = shap_result.shap_values
    
    # LIME пояснення
    lime_result = lime_explainer.explain(instance, background_data)
    lime_importance = lime_result['feature_importance']
    
    # Нормалізація до однакового масштабу
    shap_normalized = shap_values / (np.linalg.norm(shap_values) + 1e-10)
    lime_normalized = lime_importance / (np.linalg.norm(lime_importance) + 1e-10)
    
    # Кореляція Пірсона
    from scipy.stats import pearsonr
    pearson_corr, _ = pearsonr(shap_normalized, lime_normalized)
    
    # Кореляція Спірмена (ранги)
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(shap_normalized, lime_normalized)
    
    # Косинусна подібність
    cosine_sim = np.dot(shap_normalized, lime_normalized) / (
        np.linalg.norm(shap_normalized) * np.linalg.norm(lime_normalized) + 1e-10
    )
    
    return {
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'cosine_similarity': cosine_sim,
        'agreement_score': (abs(pearson_corr) + abs(spearman_corr) + abs(cosine_sim)) / 3
    }
```

---

## Інтеграція в Performance Doctor

### Структура StabilityReport

```python
@dataclass
class StabilityReport:
    """Звіт про стабільність пояснень."""
    # Метрики стабільності
    robustness_score: float  # Загальна оцінка стабільності (0-1)
    cv_score: float  # Коефіцієнт варіації
    rank_stability: float  # Стабільність рангу
    top_k_stability: float  # Стабільність топ-k
    agreement_score: float  # Узгодженість між методами
    
    # Деталі
    bootstrap_metrics: Dict[str, float]
    perturbation_metrics: Dict[str, float]
    agreement_metrics: Dict[str, float]
    
    # Статус
    is_stable: bool  # Чи є пояснення стабільним
    warnings: List[str]  # Попередження про нестабільність
```

### Оновлений PerformanceDoctor з Stability Evaluation

```python
class PerformanceDoctor:
    def __init__(self, ..., evaluate_stability: bool = True):
        # ... існуючі параметри ...
        self.evaluate_stability = evaluate_stability
        self.stability_evaluator = StabilityEvaluator()
    
    def prescribe_solution(self, ..., validate_stability: bool = True):
        # ... існуюча логіка ...
        
        # Stability evaluation
        if validate_stability and self.evaluate_stability:
            stability_report = self.stability_evaluator.evaluate(
                invariant=invariant,
                current_state=current_state,
                row_idx=row_idx,
                background_data=background_data,
                diagnostics=self.diagnostics
            )
            
            # Корекція впевненості на основі стабільності
            if stability_report.robustness_score < 0.7:
                confidence *= stability_report.robustness_score
                prescription.stability_warnings = stability_report.warnings
        
        return prescription
```

---

## Критерії валідації

### Мінімальні вимоги до стабільності

Для того, щоб пояснення вважалося **прийнятним** для використання в Production:

1. **Robustness Score ≥ 0.7**
   - Висока стабільність: ≥ 0.9
   - Середня стабільність: 0.7 - 0.9
   - Низька стабільність: < 0.7 (не прийнятно)

2. **CV Score < 0.2**
   - Коефіцієнт варіації повинен бути менше 20%

3. **Rank Stability ≥ 0.8**
   - Порядок важливості ознак повинен бути стабільним

4. **Agreement Score ≥ 0.6**
   - SHAP та LIME повинні узгоджуватися

### Правила прийняття рішень

```python
def is_explanation_acceptable(stability_report: StabilityReport) -> Tuple[bool, List[str]]:
    """
    Визначає, чи є пояснення прийнятним для використання.
    
    Returns:
    --------
    is_acceptable : bool
        Чи є пояснення прийнятним
    warnings : List[str]
        Список попереджень
    """
    warnings = []
    
    # Перевірка robustness
    if stability_report.robustness_score < 0.7:
        warnings.append(
            f"⚠️ Низька стабільність (robustness={stability_report.robustness_score:.2f}). "
            "Рекомендується збільшити кількість зразків або перевірити дані."
        )
    
    # Перевірка CV
    if stability_report.cv_score >= 0.2:
        warnings.append(
            f"⚠️ Висока варіативність (CV={stability_report.cv_score:.2f}). "
            "Пояснення може бути нестабільним."
        )
    
    # Перевірка rank stability
    if stability_report.rank_stability < 0.8:
        warnings.append(
            f"⚠️ Нестабільний порядок важливості (rank_stability={stability_report.rank_stability:.2f}). "
            "Топ-ознаки можуть змінюватися між запусками."
        )
    
    # Перевірка agreement
    if stability_report.agreement_score < 0.6:
        warnings.append(
            f"⚠️ Низька узгодженість між методами (agreement={stability_report.agreement_score:.2f}). "
            "SHAP та LIME дають різні результати."
        )
    
    is_acceptable = (
        stability_report.robustness_score >= 0.7 and
        stability_report.cv_score < 0.2 and
        stability_report.rank_stability >= 0.8 and
        stability_report.agreement_score >= 0.6
    )
    
    return is_acceptable, warnings
```

---

## Практичні рекомендації

### Коли оцінювати стабільність?

1. **Обов'язково:**
   - Перед застосуванням рекомендацій у production
   - При зміні параметрів explainer
   - При зміні структури даних

2. **Рекомендовано:**
   - Під час первинного налаштування системи
   - При періодичному моніторингу (раз на тиждень/місяць)
   - При виявленні несподіваних результатів

### Як покращити стабільність?

1. **Збільшити кількість зразків:**
   - Для SHAP: збільшити `n_samples`
   - Для LIME: збільшити `n_samples` та налаштувати `kernel_width`

2. **Використовувати більший background dataset:**
   - Більше фонових даних → стабільніші пояснення

3. **Усереднення результатів:**
   - Генерувати кілька пояснень та усереднювати

4. **Фільтрація нестабільних ознак:**
   - Виключати ознаки з високою варіативністю

---

## Висновок

**Stability Evaluation** — це обов'язковий етап валідації системи Performance Doctor. Без перевірки стабільності пояснень неможливо гарантувати надійність рекомендацій.

**Ключові висновки:**
1. ✅ Пояснення повинні бути стабільними для прийняття рішень
2. ✅ Robustness score ≥ 0.7 — мінімальна вимога
3. ✅ Узгодженість між методами (SHAP/LIME) — важливий індикатор
4. ✅ Stability evaluation інтегрована в Performance Doctor

**Наступний крок:** Використовуйте stability evaluation як частину валідації перед застосуванням рекомендацій у production.

---

## Додаткові матеріали

- **Robustness in XAI:** Alvarez-Melis, D., & Jaakkola, T. (2018). "Towards robust interpretability with self-explaining neural networks." *NeurIPS*.
- **Stability Metrics:** Yeh, C. K., et al. (2019). "On the (in)fidelity and sensitivity of explanations." *NeurIPS*.
- **Bootstrap Methods:** Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.


