---
title: "04 Geometry Of Meaning"
type: lecture
module: Семантика
prerequisites: module 3
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# Геометрія Значення: Від Слів до Векторів

"DB down" та "Connection lost" означають одне й те саме. Але для Bag of Words це різні фрази з нульовим перетином слів.

Як математика допомагає зрозуміти, що синоніми близькі, навіть якщо слова різні? Відповідь лежить у векторній алгебрі: слова стають точками в просторі, а схожість — відстанню між ними.

## Проблема Синонімів

### Приклад з Технічними Логами

**Лог 1:** "Database connection failed"
**Лог 2:** "DB down"
**Лог 3:** "Connection lost"

Для людини всі три означають критичний збій бази даних. Для Bag of Words:

$$
X_1 = \{ \texttt{"Database"}, \texttt{"connection"}, \texttt{"failed"} \}
$$

$$
X_2 = \{ \texttt{"DB"}, \texttt{"down"} \}
$$

$$
X_3 = \{ \texttt{"Connection"}, \texttt{"lost"} \}
$$

**Перетин:** $\emptyset$ (порожня множина)

**Висновок:** Bag of Words не бачить семантичної схожості.

### Чому Це Критично

У попередніх розділах ми бачили, що Naive Bayes класифікує "Connection refused" як Normal, бо "Connection" частіше в нормальних логах. Але якщо система генерує "DB unavailable" замість "Connection refused", класифікатор не зрозуміє, що це та сама проблема.

**Реальний сценарій:**

```
Критичні збої в логах:
  - "Database connection timeout"
  - "DB connection failed"
  - "Connection to database lost"
  - "Database unavailable"

Навчальні дані містили лише:
  - "Database connection timeout"
```

Naive Bayes не зрозуміє, що інші варіанти — це те саме.

## Векторні Простори: Математична Основа

### Від Множини до Вектора

**Bag of Words:** Текст → Множина слів → Вектор частот

**Embeddings:** Текст → Вектор чисел (координати в просторі)

### Формальне Визначення

**Векторний простір $\mathbb{R}^N$:**

- Кожне слово $w$ представлене як вектор $\mathbf{v}_w \in \mathbb{R}^N$
- $N$ — розмірність простору (зазвичай 100-300)
- Координати вектора кодують семантичні властивості слова

**Приклад:**

$$
\mathbf{v}_{\texttt{"database"}} = (0.2, -0.1, 0.5, \ldots, 0.3) \in \mathbb{R}^{300}
$$

$$
\mathbf{v}_{\texttt{"connection"}} = (0.3, -0.05, 0.4, \ldots, 0.25) \in \mathbb{R}^{300}
$$

### Властивості векторного простору

1. **Додавання:** $\mathbf{v}_1 + \mathbf{v}_2$ — комбінація значень
2. **Множення на скаляр:** $\alpha \mathbf{v}$ — масштабування
3. **Скалярний добуток:** $\mathbf{v}_1 \cdot \mathbf{v}_2 = \sum_{i=1}^{N} v_{1,i} \cdot v_{2,i}$
4. **Норма:** $\lVert \mathbf{v} \rVert = \sqrt{\mathbf{v} \cdot \mathbf{v}} = \sqrt{\sum_{i=1}^{N} v_i^2}$

## Косинусна відстань: міра схожості

### Євклідова відстань

**Визначення:**

$$
d(\mathbf{v}_1, \mathbf{v}_2) = \lVert \mathbf{v}_1 - \mathbf{v}_2 \rVert = \sqrt{\sum_{i=1}^{N} (v_{1,i} - v_{2,i})^2}
$$

**Проблема:** Залежить від довжини векторів. Два довгі вектори можуть бути далеко, навіть якщо вони спрямовані однаково.

### Косинусна відстань

**Ідея:** Вимірюємо кут між векторами, а не їх абсолютну відстань.

**Косинус кута:**

$$
\cos(\theta) = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\lVert \mathbf{v}_1 \rVert \cdot \lVert \mathbf{v}_2 \rVert}
$$

**Косинусна відстань:**

$$
d_{\cos}(\mathbf{v}_1, \mathbf{v}_2) = 1 - \cos(\theta) = 1 - \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\lVert \mathbf{v}_1 \rVert \cdot \lVert \mathbf{v}_2 \rVert}
$$

**Властивості:**
- $d_{\cos} \in [0, 2]$ (для векторів, що можуть мати від'ємні координати)
- $d_{\cos} = 0$ → вектори однаково спрямовані (ідеальна схожість)
- $d_{\cos} = 2$ → вектори протилежно спрямовані (максимальна відмінність)

### Нормалізація

Якщо вектори нормалізовані ($\lVert \mathbf{v} \rVert = 1$), то:

$$
\cos(\theta) = \mathbf{v}_1 \cdot \mathbf{v}_2
$$

$$
d_{\cos}(\mathbf{v}_1, \mathbf{v}_2) = 1 - \mathbf{v}_1 \cdot \mathbf{v}_2
$$

**Переваги:**
- Не залежить від довжини векторів
- Фокусується на напрямку (семантиці), а не на частоті

### Приклад

**Вектори:**

$$
\mathbf{v}_{\texttt{"database"}} = (0.6, 0.8)
$$

$$
\mathbf{v}_{\texttt{"DB"}} = (0.3, 0.4)
$$

$$
\mathbf{v}_{\texttt{"connection"}} = (0.8, 0.6)
$$

**Косинусна схожість:**

$$
\cos(\texttt{"database"}, \texttt{"DB"}) = \frac{0.6 \cdot 0.3 + 0.8 \cdot 0.4}{\sqrt{0.6^2 + 0.8^2} \cdot \sqrt{0.3^2 + 0.4^2}} = \frac{0.18 + 0.32}{1.0 \cdot 0.5} = 1.0
$$

**Висновок:** `database` та `DB` майже ідентичні за напрямком (синоніми).

$$
\cos(\texttt{"database"}, \texttt{"connection"}) = \frac{0.6 \cdot 0.8 + 0.8 \cdot 0.6}{1.0 \cdot 1.0} = \frac{0.48 + 0.48}{1.0} = 0.96
$$

**Висновок:** `database` та `connection` теж близькі (пов'язані концепти).

## Word2Vec: Інтуїція та Математика

### Distributional Hypothesis

**Гіпотеза:** Слова, які з'являються в подібних контекстах, мають подібні значення.

**Приклад:**
- "Database" з'являється з "connection", "timeout", "query"
- "DB" з'являється з "connection", "timeout", "query"
- → "Database" та "DB" мають подібні вектори

### Skip-gram Модель

**Ідея:** Навчаємо модель передбачати контекстні слова за заданим словом.

**Формалізація:**

Для слова $w$ та контекстного слова $c$:

$$
P(c \mid w) = \frac{\exp(\mathbf{v}_c \cdot \mathbf{v}_w)}{\sum_{c' \in V} \exp(\mathbf{v}_{c'} \cdot \mathbf{v}_w)}
$$

де:
- $\mathbf{v}_w$ — вектор слова (word embedding)
- $\mathbf{v}_c$ — вектор контексту (context embedding)
- $V$ — словник

**Мета:** Максимізувати ймовірність справжніх контекстних слів.

### Відоме Рівняння: King - Man + Woman = Queen

**Інтуїція:** Векторні різниці кодують семантичні відношення.

**Математика:**

Якщо:

$$
\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} \approx \mathbf{v}_{\text{Queen}} - \mathbf{v}_{\text{Woman}}
$$

То:

$$
\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} \approx \mathbf{v}_{\text{Queen}}
$$

**Чому це працює:**

Векторна різниця $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}}$ кодує відношення "royalty" (королівськість). Додавання $\mathbf{v}_{\text{Woman}}$ дає "royal woman" = Queen.

**Для технічних термінів:**

$$
\mathbf{v}_{\texttt{"database"}} - \mathbf{v}_{\texttt{"connection"}} + \mathbf{v}_{\texttt{"server"}} \approx \mathbf{v}_{\texttt{"DB server"}}
$$

### Continuous Bag of Words (CBOW)

**Альтернатива Skip-gram:** Передбачає слово за контекстом.

**Формалізація:**

$$
P(w \mid c_1, c_2, \ldots, c_n) = \frac{\exp(\mathbf{v}_w \cdot \bar{\mathbf{v}}_c)}{\sum_{w' \in V} \exp(\mathbf{v}_{w'} \cdot \bar{\mathbf{v}}_c)}
$$

де $\bar{\mathbf{v}}_c = \frac{1}{n} \sum_{i=1}^{n} \mathbf{v}_{c_i}$ — середній вектор контексту.

## Реалізація: Обчислення Схожості

```python
"""
Реалізація обчислення косинусної відстані та семантичної схожості.
Демонструє перехід від Bag of Words до векторних представлень.
"""

from typing import List, Tuple, Dict
import numpy as np
from dataclasses import dataclass
from collections import Counter


@dataclass
class SimilarityResult:
    """Результат обчислення схожості."""
    word1: str
    word2: str
    cosine_similarity: float
    cosine_distance: float
    euclidean_distance: float


class WordEmbedding:
    """
    Простий клас для роботи з векторними представленнями слів.
    """
    
    def __init__(self, word: str, vector: np.ndarray):
        """
        Args:
            word: Слово
            vector: Вектор представлення (numpy array)
        """
        self.word = word
        self.vector = vector
        self.normalized_vector = self._normalize(vector)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Нормалізує вектор до одиничної довжини."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def cosine_similarity(self, other: 'WordEmbedding') -> float:
        """
        Обчислює косинусну схожість з іншим словом.
        
        Returns:
            Значення в діапазоні [-1, 1] (для нормалізованих векторів)
        """
        return np.dot(self.normalized_vector, other.normalized_vector)
    
    def cosine_distance(self, other: 'WordEmbedding') -> float:
        """
        Обчислює косинусну відстань.
        
        Returns:
            Значення в діапазоні [0, 2]
        """
        return 1.0 - self.cosine_similarity(other)
    
    def euclidean_distance(self, other: 'WordEmbedding') -> float:
        """Обчислює євклідову відстань."""
        return np.linalg.norm(self.vector - other.vector)
    
    def similarity_to(self, other: 'WordEmbedding') -> SimilarityResult:
        """Повний аналіз схожості."""
        return SimilarityResult(
            word1=self.word,
            word2=other.word,
            cosine_similarity=self.cosine_similarity(other),
            cosine_distance=self.cosine_distance(other),
            euclidean_distance=self.euclidean_distance(other)
        )


class SimpleEmbeddingSpace:
    """
    Простий простір embeddings для демонстрації концепцій.
    У реальності використовуються попередньо навчені моделі (Word2Vec, GloVe).
    """
    
    def __init__(self, dimension: int = 100):
        self.dimension = dimension
        self.embeddings: Dict[str, WordEmbedding] = {}
    
    def add_word(self, word: str, vector: np.ndarray) -> None:
        """Додає слово з його векторним представленням."""
        if len(vector) != self.dimension:
            raise ValueError(f"Вектор має мати розмірність {self.dimension}")
        self.embeddings[word] = WordEmbedding(word, vector)
    
    def get_embedding(self, word: str) -> WordEmbedding:
        """Отримує embedding для слова."""
        if word not in self.embeddings:
            raise KeyError(f"Слово '{word}' не знайдено в просторі")
        return self.embeddings[word]
    
    def find_most_similar(
        self, 
        word: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Знаходить найбільш схожі слова.
        
        Returns:
            Список пар (слово, косинусна схожість), відсортований за спаданням
        """
        if word not in self.embeddings:
            return []
        
        target = self.get_embedding(word)
        similarities = []
        
        for other_word, other_embedding in self.embeddings.items():
            if other_word == word:
                continue
            similarity = target.cosine_similarity(other_embedding)
            similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analogy(
        self, 
        word1: str, 
        word2: str, 
        word3: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Виконує аналогію: word1 - word2 + word3 = ?
        
        Приклад: "king" - "man" + "woman" = "queen"
        """
        if not all(w in self.embeddings for w in [word1, word2, word3]):
            return []
        
        v1 = self.get_embedding(word1).vector
        v2 = self.get_embedding(word2).vector
        v3 = self.get_embedding(word3).vector
        
        # Обчислюємо цільовий вектор
        target_vector = v1 - v2 + v3
        target_embedding = WordEmbedding("target", target_vector)
        
        # Шукаємо найближчі слова
        similarities = []
        for word, embedding in self.embeddings.items():
            if word in [word1, word2, word3]:
                continue
            similarity = target_embedding.cosine_similarity(embedding)
            similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_demo_embeddings() -> SimpleEmbeddingSpace:
    """
    Створює демонстраційний простір embeddings.
    У реальності використовуються попередньо навчені моделі.
    """
    np.random.seed(42)
    space = SimpleEmbeddingSpace(dimension=50)
    
    # Синоніми мають подібні вектори
    # "database" та "DB" — дуже схожі
    base_db = np.random.randn(50)
    space.add_word("database", base_db)
    space.add_word("DB", base_db + 0.1 * np.random.randn(50))  # Невеликий шум
    
    # "connection" пов'язане з "database"
    base_conn = base_db + 0.3 * np.random.randn(50)
    space.add_word("connection", base_conn)
    space.add_word("conn", base_conn + 0.1 * np.random.randn(50))
    
    # "failed" та "lost" — синоніми в контексті помилок
    base_fail = np.random.randn(50)
    space.add_word("failed", base_fail)
    space.add_word("lost", base_fail + 0.2 * np.random.randn(50))
    space.add_word("down", base_fail + 0.15 * np.random.randn(50))
    
    # "timeout" пов'язане з "failed"
    space.add_word("timeout", base_fail + 0.4 * np.random.randn(50))
    
    # Непов'язані слова
    space.add_word("successful", np.random.randn(50))
    space.add_word("ready", np.random.randn(50))
    space.add_word("established", np.random.randn(50))
    
    return space


def demonstrate_embeddings() -> None:
    """Демонструє роботу з embeddings та косинусною відстанню."""
    print("=" * 70)
    print("ГЕОМЕТРІЯ ЗНАЧЕННЯ: ВІД СЛІВ ДО ВЕКТОРІВ")
    print("=" * 70)
    print()
    
    space = create_demo_embeddings()
    
    # Тест 1: Синоніми
    print("ТЕСТ 1: Синоніми")
    print("-" * 70)
    word1 = space.get_embedding("database")
    word2 = space.get_embedding("DB")
    result = word1.similarity_to(word2)
    
    print(f"Слова: '{result.word1}' vs '{result.word2}'")
    print(f"  Косинусна схожість: {result.cosine_similarity:.4f}")
    print(f"  Косинусна відстань: {result.cosine_distance:.4f}")
    print(f"  Євклідова відстань: {result.euclidean_distance:.4f}")
    print()
    
    # Тест 2: Пов'язані концепти
    print("ТЕСТ 2: Пов'язані концепти")
    print("-" * 70)
    word1 = space.get_embedding("database")
    word2 = space.get_embedding("connection")
    result = word1.similarity_to(word2)
    
    print(f"Слова: '{result.word1}' vs '{result.word2}'")
    print(f"  Косинусна схожість: {result.cosine_similarity:.4f}")
    print(f"  Косинусна відстань: {result.cosine_distance:.4f}")
    print()
    
    # Тест 3: Найбільш схожі слова
    print("ТЕСТ 3: Найбільш схожі слова до 'database'")
    print("-" * 70)
    similar = space.find_most_similar("database", top_k=5)
    for word, similarity in similar:
        print(f"  '{word}': {similarity:.4f}")
    print()
    
    # Тест 4: Аналогія
    print("ТЕСТ 4: Аналогія (database - connection + timeout = ?)")
    print("-" * 70)
    analogy_results = space.analogy("database", "connection", "timeout", top_k=3)
    for word, similarity in analogy_results:
        print(f"  '{word}': {similarity:.4f}")
    print()
    
    # Тест 5: Порівняння з Bag of Words
    print("ТЕСТ 5: Порівняння з Bag of Words")
    print("-" * 70)
    print("Bag of Words:")
    print("  'database connection failed' ∩ 'DB down' = ∅")
    print("  → Схожість: 0")
    print()
    print("Embeddings:")
    db = space.get_embedding("DB")
    failed = space.get_embedding("failed")
    down = space.get_embedding("down")
    
    # Середній вектор фрази
    phrase1_vector = (db.vector + failed.vector) / 2
    phrase2_vector = (db.vector + down.vector) / 2
    
    phrase1_embedding = WordEmbedding("phrase1", phrase1_vector)
    phrase2_embedding = WordEmbedding("phrase2", phrase2_vector)
    
    similarity = phrase1_embedding.cosine_similarity(phrase2_embedding)
    print(f"  'DB failed' vs 'DB down': {similarity:.4f}")
    print()
    
    print("=" * 70)
    print("ВИСНОВОК:")
    print("  Embeddings дозволяють знаходити семантичну схожість,")
    print("  навіть коли слова різні. Це критично для технічних логів,")
    print("  де одна проблема може бути описана різними термінами.")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_embeddings()
```

## Застосування до Технічних Логів

### Проблема: Різні Формули, Одна Проблема

**Критичні збої можуть бути описані як:**
- "Database connection failed"
- "DB connection timeout"
- "Connection to database lost"
- "Database unavailable"

**Bag of Words:** Всі різні (перетин = ∅)

**Embeddings:** Всі схожі (косинусна схожість > 0.8)

### Рішення: Класифікація через середній вектор

Для фрази $d = \{w_1, w_2, \ldots, w_n\}$:

$$
\bar{\mathbf{v}}_d = \frac{1}{n} \sum_{i=1}^{n} \mathbf{v}_{w_i}
$$

**Класифікація:** Порівнюємо $\bar{\mathbf{v}}_d$ з прототипами класів.

**Приклад:**

$$
\bar{\mathbf{v}}_{\texttt{"DB down"}} \approx \bar{\mathbf{v}}_{\texttt{"database connection failed"}}
$$

Обидва класифікуються як Critical.

## Ключові Висновки

1. **Векторні простори:** Слова стають точками в $\mathbb{R}^N$, де $N$ — розмірність embeddings.

2. **Косинусна відстань:** Вимірює семантичну схожість через напрямок векторів, а не їх довжину.

3. **Word2Vec:** Навчає вектори через передбачення контексту, кодуючи семантичні відношення.

4. **Синоніми близькі:** "DB" та "database" мають високу косинусну схожість, навіть якщо слова різні.

5. **Контекст важливий:** Embeddings враховують контекст використання, на відміну від Bag of Words.

У наступному розділі ми подивимося, як трансформери (BERT) використовують embeddings та механізм attention для ще кращого розуміння контексту.

## Рекомендована Література

### Класичні Роботи про Word Embeddings

1. **Mikolov, T., et al.** (2013). "Efficient estimation of word representations in vector space"
   - arXiv:1301.3781. Оригінальна робота про Word2Vec (Skip-gram та CBOW).

2. **Mikolov, T., et al.** (2013). "Distributed representations of words and phrases and their compositionality"
   - NIPS. Розширення Word2Vec, включаючи відоме рівняння King - Man + Woman = Queen.

3. **Pennington, J., Socher, R., & Manning, C. D.** (2014). "GloVe: Global Vectors for Word Representation"
   - EMNLP. Альтернативний підхід до Word2Vec через глобальну статистику.

### Distributional Hypothesis та Семантика

4. **Harris, Z. S.** (1954). "Distributional structure"
   - Word, 10(2-3), 146-162. Оригінальна формулювання distributional hypothesis.

5. **Turney, P. D., & Pantel, P.** (2010). "From frequency to meaning: Vector space models of semantics"
   - Journal of Artificial Intelligence Research, 37, 141-188. Огляд векторних моделей семантики.

### Математична Теорія

6. **Levy, O., & Goldberg, Y.** (2014). "Neural word embedding as implicit matrix factorization"
   - NIPS. Математичне обґрунтування Word2Vec через матричну факторизацію.

7. **Arora, S., et al.** (2016). "A latent variable model approach to pmi-based word embeddings"
   - TACL. Теоретичне обґрунтування embeddings через latent variable models.

### Практична Реалізація

8. **Řehůřek, R., & Sojka, P.** (2010). "Software Framework for Topic Modelling with Large Corpora"
   - LREC Workshop. Gensim library для Word2Vec.

9. **Pedregosa, F., et al.** (2011). "Scikit-learn: Machine Learning in Python"
   - Документація: https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity

### Від Word2Vec до BERT

10. **Peters, M. E., et al.** (2018). "Deep contextualized word representations"
    - NAACL. ELMo — перший крок до контекстних embeddings.

11. **Devlin, J., et al.** (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    - NAACL. BERT — контекстні embeddings через трансформери.

---

**Примітка для студентів:** Почніть з Mikolov et al. (2013) для розуміння Word2Vec, потім перейдіть до Levy & Goldberg для математичної строгості. Для практики використовуйте Gensim та приклади з Turney & Pantel. Перед переходом до BERT прочитайте Peters et al. про ELMo.
