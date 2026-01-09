---
title: "Семінар А: Геометрія Високих Розмірностей та Візуалізація"
layout: default
author: Віталій Платонов
---

# Семінар А: Геометрія Високих Розмірностей та Візуалізація

BERT генерує вектори з 768 розмірностями. Як уявити собі простір з 768 вимірами? Чому "Database errors" та "Network timeouts" утворюють окремі кластери, навіть якщо вони обидва критичні?

Цей семінар заповнює прогалину між теорією векторних просторів та практичним розумінням того, як моделі "бачать" структуру даних у високих розмірностях.

## Чому Це Важливо?

### Проблема Високих Розмірностей

**BERT-base:** Кожен текст перетворюється на вектор з 768 координатами.

$$\mathbf{v}_{\text{log}} = (v_1, v_2, \ldots, v_{768}) \in \mathbb{R}^{768}$$

**Питання:** Як уявити собі простір з 768 вимірами?

**Відповідь:** Ми не можемо. Людський мозок еволюційно пристосований до 3D простору. Але математика дозволяє працювати з високими розмірностями та візуалізувати структуру через зниження розмірності.

### Реальний Приклад: Кластеризація Логів

**Сценарій:** У нас є 10,000 логів, кожен представлений як 768-вимірний вектор від BERT.

**Питання:** Чи можна побачити, що:
- Логи "Database connection timeout" утворюють один кластер?
- Логи "Network timeout" утворюють інший кластер?
- Логи "Successful request" утворюють третій кластер?

**Відповідь:** Так, але тільки після зниження розмірності до 2D або 3D для візуалізації.

## Теорія: Прокляття Високих Розмірностей

### Проблема Розрідженості

У високих розмірностях більшість точок знаходяться на "краях" простору, а не в центрі.

**Приклад:** У 768-вимірному просторі, якщо ми випадково виберемо дві точки, вони майже завжди будуть на однаковій відстані одна від одної.

**Наслідок:** Класична євклідова відстань втрачає сенс у високих розмірностях.

### Відстань у Високих Розмірностях

**Теорема:** У $d$-вимірному просторі, якщо $d \to \infty$, то:

$$\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

де $d_{\max}$ та $d_{\min}$ — максимальна та мінімальна відстань між точками.

**Інтуїція:** У високих розмірностях всі відстані стають подібними. Це робить кластеризацію та пошук найближчих сусідів складнішими.

**Рішення:** Використовувати косинусну відстань (як ми бачили в розділі 04) або методи зниження розмірності для візуалізації.

## Методи Зниження Розмірності

### Principal Component Analysis (PCA)

**Ідея:** Знайти напрямки максимальної варіації даних та спроектувати на них.

**Математика:**

Для датасету $\mathbf{X} \in \mathbb{R}^{n \times d}$ (n зразків, d ознак):

1. Центруємо дані: $\mathbf{X}_{\text{centered}} = \mathbf{X} - \bar{\mathbf{X}}$
2. Обчислюємо коваріаційну матрицю: $\mathbf{C} = \frac{1}{n-1} \mathbf{X}_{\text{centered}}^T \mathbf{X}_{\text{centered}}$
3. Знаходимо власні вектори $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ матриці $\mathbf{C}$ (напрямки максимальної варіації)
4. Проектуємо: $\mathbf{X}_{\text{reduced}} = \mathbf{X}_{\text{centered}} \cdot [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k]$

**Обмеження:** PCA — лінійний метод. Він не зберігає локальні структури (наприклад, кластери).

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Ідея:** Зберігати локальні відстані між точками при зниженні розмірності.

**Алгоритм:**

1. **Високі розмірності:** Для кожної пари точок $(i, j)$ обчислюємо ймовірність $p_{ij}$, що точка $j$ є "сусідом" точки $i$:

$$p_{ij} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

2. **Низькі розмірності:** Створюємо подібні ймовірності $q_{ij}$ у 2D просторі:

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq i} (1 + \|y_i - y_k\|^2)^{-1}}$$

3. **Мінімізація:** Мінімізуємо розбіжність Кульбака-Лейблера між $p_{ij}$ та $q_{ij}$:

$$KL(P \| Q) = \sum_{i} \sum_{j \neq i} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**Переваги:**
- Зберігає локальні структури (кластери)
- Добре працює для візуалізації

**Недоліки:**
- Глобальні відстані можуть бути спотворені
- Результат залежить від параметра `perplexity`
- Обчислювально дорогий для великих датасетів

### UMAP (Uniform Manifold Approximation and Projection)

**Ідея:** Припустити, що дані лежать на многовиді (manifold) у високих розмірностях та зберегти топологію цього многовида.

**Математика:**

1. **Побудова графа сусідів:** Створюємо зважений граф, де ваги ребер відображають відстані між сусідами.

2. **Оптимізація:** Мінімізуємо cross-entropy між розподілами у високих та низьких розмірностях:

$$\text{CE}(P, Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \log \frac{1 - p_{ij}}{1 - q_{ij}}$$

**Переваги:**
- Швидший за t-SNE
- Краще зберігає глобальну структуру
- Менше залежить від параметрів

**Недоліки:**
- Складніша математика (теорія многовидів)

## Практика: Візуалізація Кластерів Логів

### Підготовка Даних

```python
"""
Візуалізація BERT embeddings для технічних логів.
Демонструє, як вектори логів кластеруються за типами інцидентів.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from transformers import BertTokenizer, BertModel
import torch
from typing import List, Tuple
import seaborn as sns

# Налаштування для красивих графіків
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
```

### Генерація BERT Embeddings

```python
class BERTEmbeddingExtractor:
    """
    Клас для витягування BERT embeddings з текстів.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Args:
            model_name: Назва попередньо навченої BERT моделі
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_embeddings(self, texts: List[str], max_length: int = 128) -> np.ndarray:
        """
        Витягує BERT embeddings для списку текстів.
        
        Args:
            texts: Список текстів (логів)
            max_length: Максимальна довжина послідовності
        
        Returns:
            Масив embeddings розміру (n_texts, 768)
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Токенізація
                encoded = self.tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Перенесення на пристрій
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Отримання embeddings
                outputs = self.model(**encoded)
                
                # Використовуємо [CLS] токен (перший токен) як представлення всього тексту
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding[0])
        
        return np.array(embeddings)
```

### Створення Синтетичного Датасету Логів

```python
def generate_log_dataset() -> Tuple[List[str], List[str]]:
    """
    Генерує синтетичний датасет логів з різними типами інцидентів.
    
    Returns:
        (texts, labels) - списки текстів та їх міток
    """
    texts = []
    labels = []
    
    # Database errors
    db_errors = [
        "Database connection timeout after 30 seconds",
        "Failed to connect to database server",
        "Database connection pool exhausted",
        "SQL query execution timeout",
        "Database deadlock detected",
        "Cannot establish database connection",
        "Database server is not responding",
        "Connection to database lost"
    ]
    
    # Network timeouts
    network_errors = [
        "Network timeout: connection refused",
        "Failed to establish network connection",
        "Network interface is down",
        "Connection timeout to remote server",
        "Network packet loss detected",
        "Unable to reach network endpoint",
        "Network connection reset by peer",
        "DNS resolution timeout"
    ]
    
    # Authentication failures
    auth_errors = [
        "Authentication failed: invalid credentials",
        "User login attempt failed",
        "Access denied: insufficient permissions",
        "Authentication token expired",
        "Failed to verify user identity",
        "Invalid authentication token",
        "User account is locked",
        "Authentication service unavailable"
    ]
    
    # Successful operations
    success_logs = [
        "Request processed successfully",
        "Operation completed without errors",
        "Transaction committed successfully",
        "User authentication successful",
        "Data synchronization completed",
        "Backup operation finished successfully",
        "Service started successfully",
        "Health check passed"
    ]
    
    # Додаємо дані
    for text in db_errors:
        texts.append(text)
        labels.append("Database Error")
    
    for text in network_errors:
        texts.append(text)
        labels.append("Network Timeout")
    
    for text in auth_errors:
        texts.append(text)
        labels.append("Authentication Failure")
    
    for text in success_logs:
        texts.append(text)
        labels.append("Success")
    
    return texts, labels
```

### Візуалізація з PCA

```python
def visualize_with_pca(embeddings: np.ndarray, labels: List[str]) -> None:
    """
    Візуалізує embeddings за допомогою PCA.
    
    Args:
        embeddings: BERT embeddings (n_samples, 768)
        labels: Мітки класів
    """
    print("Застосовуємо PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Створюємо графік
    plt.figure(figsize=(10, 8))
    
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=label,
            c=[colors[i]],
            alpha=0.7,
            s=100
        )
    
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA Visualization of BERT Embeddings for Log Clustering')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    print("Графік збережено: pca_visualization.png")
    plt.show()
    
    # Виводимо статистику
    print(f"\nPCA Statistics:")
    print(f"  Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2 variance: {pca.explained_variance_ratio_[1]:.2%}")
```

### Візуалізація з t-SNE

```python
def visualize_with_tsne(embeddings: np.ndarray, labels: List[str], perplexity: int = 30) -> None:
    """
    Візуалізує embeddings за допомогою t-SNE.
    
    Args:
        embeddings: BERT embeddings (n_samples, 768)
        labels: Мітки класів
        perplexity: Параметр t-SNE (рекомендовано: 5-50)
    """
    print(f"Застосовуємо t-SNE (perplexity={perplexity})...")
    print("Це може зайняти час...")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Створюємо графік
    plt.figure(figsize=(10, 8))
    
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=label,
            c=[colors[i]],
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidths=0.5
        )
    
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f't-SNE Visualization of BERT Embeddings (perplexity={perplexity})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    print("Графік збережено: tsne_visualization.png")
    plt.show()
```

### Візуалізація з UMAP

```python
def visualize_with_umap(embeddings: np.ndarray, labels: List[str], n_neighbors: int = 15) -> None:
    """
    Візуалізує embeddings за допомогою UMAP.
    
    Args:
        embeddings: BERT embeddings (n_samples, 768)
        labels: Мітки класів
        n_neighbors: Кількість сусідів для побудови графа
    """
    print(f"Застосовуємо UMAP (n_neighbors={n_neighbors})...")
    
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Створюємо графік
    plt.figure(figsize=(10, 8))
    
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=label,
            c=[colors[i]],
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidths=0.5
        )
    
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'UMAP Visualization of BERT Embeddings (n_neighbors={n_neighbors})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('umap_visualization.png', dpi=300, bbox_inches='tight')
    print("Графік збережено: umap_visualization.png")
    plt.show()
```

### Порівняння Методів

```python
def compare_dimension_reduction_methods(embeddings: np.ndarray, labels: List[str]) -> None:
    """
    Порівнює PCA, t-SNE та UMAP на одному графіку.
    
    Args:
        embeddings: BERT embeddings (n_samples, 768)
        labels: Мітки класів
    """
    print("Порівняння методів зниження розмірності...")
    
    # Застосовуємо всі методи
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings)
    
    # Створюємо графік з трьома підграфіками
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    unique_labels = list(set(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    results = [
        (embeddings_pca, "PCA", axes[0]),
        (embeddings_tsne, "t-SNE", axes[1]),
        (embeddings_umap, "UMAP", axes[2])
    ]
    
    for embeddings_2d, method_name, ax in results:
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=label,
                c=[colors[i]],
                alpha=0.7,
                s=80,
                edgecolors='black',
                linewidths=0.3
            )
        
        ax.set_title(method_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_dimension_reduction.png', dpi=300, bbox_inches='tight')
    print("Графік збережено: comparison_dimension_reduction.png")
    plt.show()
```

### Головна Функція

```python
def main():
    """
    Головна функція для демонстрації візуалізації кластерів логів.
    """
    print("=" * 70)
    print("СЕМІНАР А: ГЕОМЕТРІЯ ВИСОКИХ РОЗМІРНОСТЕЙ ТА ВІЗУАЛІЗАЦІЯ")
    print("=" * 70)
    print()
    
    # Генеруємо датасет
    print("1. Генерація синтетичного датасету логів...")
    texts, labels = generate_log_dataset()
    print(f"   Згенеровано {len(texts)} логів з {len(set(labels))} класами")
    print()
    
    # Витягуємо BERT embeddings
    print("2. Витягування BERT embeddings...")
    print("   Це може зайняти час (завантаження моделі)...")
    extractor = BERTEmbeddingExtractor()
    embeddings = extractor.extract_embeddings(texts)
    print(f"   Отримано embeddings розміру: {embeddings.shape}")
    print(f"   Розмірність кожного вектора: {embeddings.shape[1]} (BERT-base)")
    print()
    
    # Візуалізуємо з різними методами
    print("3. Візуалізація з PCA...")
    visualize_with_pca(embeddings, labels)
    print()
    
    print("4. Візуалізація з t-SNE...")
    visualize_with_tsne(embeddings, labels, perplexity=30)
    print()
    
    print("5. Візуалізація з UMAP...")
    try:
        visualize_with_umap(embeddings, labels, n_neighbors=15)
    except ImportError:
        print("   Помилка: UMAP не встановлено. Встановіть: pip install umap-learn")
    print()
    
    print("6. Порівняння методів...")
    try:
        compare_dimension_reduction_methods(embeddings, labels)
    except ImportError:
        print("   Помилка: UMAP не встановлено. Пропускаємо порівняння.")
    print()
    
    print("=" * 70)
    print("ВИСНОВКИ:")
    print("=" * 70)
    print("1. BERT embeddings мають 768 розмірностей - неможливо візуалізувати напряму")
    print("2. PCA - швидкий, але лінійний метод (може втрачати кластери)")
    print("3. t-SNE - зберігає локальні структури, але спотворює глобальні відстані")
    print("4. UMAP - балансує між локальною та глобальною структурою")
    print("5. Всі методи показують кластеризацію логів за типами інцидентів")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

## Аналіз Результатів

### Що Показують Графіки?

**PCA:**
- Може показувати загальну структуру, але кластери можуть бути змішані
- Швидкий метод, але втрачає нелінійні залежності

**t-SNE:**
- Чітко розділяє кластери (Database errors, Network timeouts, тощо)
- Кластери розташовані окремо, що підтверджує, що BERT розрізняє типи інцидентів
- Може спотворювати глобальні відстані (кластери можуть бути ближче/далі, ніж насправді)

**UMAP:**
- Зберігає як локальну, так і глобальну структуру
- Кластери чіткі, але також видно загальну структуру даних
- Швидший за t-SNE для великих датасетів

### Практичне Застосування

**1. Діагностика Моделі:**
- Якщо кластери змішані → модель не розрізняє типи інцидентів
- Якщо кластери чіткі → модель успішно кодує семантичні відмінності

**2. Виявлення Аномалій:**
- Точки, які не входять до жодного кластера → потенційні аномалії
- Нові типи помилок, які модель не бачила під час навчання

**3. Оцінка Якості Embeddings:**
- Якщо схожі логи (наприклад, обидва про database) утворюють один кластер → embeddings якісні
- Якщо схожі логи розкидані → embeddings потребують покращення

## Ключові Висновки

1. **Високі розмірності невидимі:** 768-вимірний простір неможливо уявити, але математика дозволяє працювати з ним.

2. **Зниження розмірності необхідне:** Для візуалізації та інтуїтивного розуміння структури даних.

3. **Різні методи для різних цілей:**
   - **PCA:** Швидкий огляд загальної структури
   - **t-SNE:** Детальна візуалізація локальних кластерів
   - **UMAP:** Баланс між локальною та глобальною структурою

4. **Кластеризація підтверджує якість моделі:** Якщо BERT embeddings утворюють чіткі кластери за типами інцидентів, це означає, що модель успішно кодує семантичні відмінності.

5. **Візуалізація — інструмент діагностики:** Дозволяє швидко оцінити, чи працює модель правильно, без необхідності запускати повну класифікацію.

## Рекомендована Література

### Теорія Високих Розмірностей

1. **Bellman, R.** (1961). "Adaptive Control Processes: A Guided Tour"
   - Princeton University Press. Введення концепції "curse of dimensionality".

2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). "The Elements of Statistical Learning"
   - 2nd Edition. Springer. Розділ 14: "Unsupervised Learning" (PCA та інші методи).

### t-SNE

3. **van der Maaten, L., & Hinton, G.** (2008). "Visualizing Data using t-SNE"
   - Journal of Machine Learning Research, 9, 2579-2605.
   - Оригінальна робота про t-SNE.

4. **Wattenberg, M., Viégas, F., & Johnson, I.** (2016). "How to Use t-SNE Effectively"
   - Distill. Практичний гайд: https://distill.pub/2016/misread-tsne/

### UMAP

5. **McInnes, L., Healy, J., & Melville, J.** (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"
   - arXiv:1802.03426. Оригінальна робота про UMAP.

6. **McInnes, L., Healy, J., Saul, N., & Großberger, L.** (2018). "UMAP: Uniform Manifold Approximation and Projection"
   - Journal of Open Source Software, 3(29), 861.
   - Документація: https://umap-learn.readthedocs.io/

### Практичні Застосування

7. **Kobak, D., & Berens, P.** (2019). "The art of using t-SNE for single-cell transcriptomics"
   - Nature Communications, 10(1), 1-14.
   - Практичні поради щодо використання t-SNE.

8. **Becht, E., et al.** (2019). "Dimensionality reduction for visualizing single-cell data using UMAP"
   - Nature Biotechnology, 37(1), 38-44.
   - Порівняння UMAP з t-SNE для біологічних даних.

---

**Примітка для студентів:** Почніть з практичної реалізації коду вище. Для розуміння теорії прочитайте van der Maaten & Hinton (2008) про t-SNE та McInnes et al. (2018) про UMAP. Для розуміння проблеми високих розмірностей дивіться розділ 14 у Hastie et al. (2009). Практичні поради щодо використання t-SNE дивіться у Wattenberg et al. (2016).

