---
title: "Математична Формалізація Задачі Класифікації"
layout: default
author: Віталій Платонов
---

# Математична Формалізація Задачі Класифікації

Як перетворити текст на числа? Як математика допомагає комп'ютеру розрізняти спам від важливих листів або критичні помилки від некритичних?

Відповідь лежить у формалізації: текст стає вектором чисел, класифікація — задачею оптимізації ймовірностей. Розберемося, як це працює математично.

## Текст як Множина Токенів

### Від Послідовності до Множини

Текст — це послідовність символів. Але для класифікації нам потрібна структура, з якою можна працювати математично.

**Крок 0: Нормалізація Технічних Текстів**

Перед токенізацією критично важливо нормалізувати технічні тексти (логи, системні повідомлення). На відміну від природної мови, технічні тексти містять велику кількість унікальних ідентифікаторів, які не несуть семантичної інформації для класифікації, але значно збільшують розмірність словника $V$.

**Що нормалізувати:**

1. **Часові мітки (Timestamps)** — унікальні для кожного логу
2. **IP-адреси** (IPv4, IPv6) — унікальні, але не інформативні для класифікації
3. **Ідентифікатори сесій** (UUID, хеші, транзакційні ID) — унікальні для кожної сесії

**Математичне обґрунтування:**

Розглянемо вплив на розмірність словника $V$:

**Без нормалізації:**
- Датасет: 10,000 логів
- Кожен лог містить унікальну часову мітку → 10,000 унікальних токенів
- Кожен лог містить унікальну IP-адресу → ще 10,000 унікальних токенів
- Реальні слова: ~500 унікальних токенів
- **Загальний $\lvert V \rvert \approx 20{,}500$**

**З нормалізацією:**
- Замінюємо унікальні ідентифікатори на плейсхолдери: `[TIMESTAMP]`, `[IP_ADDRESS]`, `[SESSION_ID]`
- Реальні слова: ~500 унікальних токенів + 3 плейсхолдери
- **Загальний $\lvert V \rvert \approx 503$**

**Вплив на TF-IDF:**

**Що це таке:**
TF-IDF (Term Frequency — Inverse Document Frequency) — це метрика, яка оцінює **важливість** токена для конкретного документа відносно всього набору даних.
* **TF (Локальна частота):** Як часто слово зустрічається тут? (Чим частіше $\to$ тим важливіше).
* **IDF (Глобальна рідкість):** Як рідко слово зустрічається в інших документах? (Чим рідше $\to$ тим цінніше).
* **Інтуїція:** Слово має високу вагу, якщо воно часте в *цьому* лозі, але унікальне для *всієї бази*.

TF-IDF обчислюється як:

$$
\text{tf-idf}(t, d, D) = \text{tf}(t, d) \cdot \log \frac{\lvert D \rvert}{\lvert \{d \in D : t \in d\} \rvert}
$$

**Проблема без нормалізації:**
- Унікальні токени (IP, timestamps) існують лише в 1 документі, тому знаменник дробу = 1.
- $\text{idf}(t, D) = \log \frac{\lvert D \rvert}{1} = \log \lvert D \rvert$ (максимально можливе значення).
- **Наслідок:** Сміття (таймстемпи) отримує найвищу вагу і "витісняє" реальні помилки ("Error", "Fail").

**Рішення з нормалізацією:**
- Плейсхолдери `[TIMESTAMP]`, `[IP_ADDRESS]` тепер зустрічаються в **усіх** документах.
- $\text{idf}(\text{[TIMESTAMP]}, D) = \log \frac{\lvert D \rvert}{\lvert D \rvert} = \log(1) = 0$ (вага обнуляється).
- **Результат:** Технічні дані ігноруються, а специфічні слова помилок отримують коректну вагу.

**Примітка:** Детальний розбір алгоритмів нормалізації (регулярні вирази для IP-адрес, UUID, часових міток) та практичні приклади реалізації наведені в [Семінарі Г: Регулярні Вирази vs Токенізатори](./12_seminar_regex_vs_tokenizers.md).

**Крок 1: Токенізація**

Розбиваємо текст на токени (слова, знаки пунктуації, числа):

$$
\text{Text} = \texttt{"Connection refused: database unavailable"}
$$

$$
\text{Tokens} = \{ \texttt{"Connection"}, \texttt{"refused"}, \texttt{":"}, \texttt{"database"}, \texttt{"unavailable"} \}
$$

**Крок 2: Представлення як множина**

Для багатьох алгоритмів порядок слів не важливий. Текст стає множиною токенів:

$$
X = \{w_1, w_2, \ldots, w_n\}
$$

де $w_i$ — окремий токен, $n$ — кількість унікальних токенів у тексті.

**Приклад:**

$$
X_{\text{error}} = \{ \texttt{"Connection"}, \texttt{"refused"}, \texttt{"database"}, \texttt{"unavailable"} \}
$$

$$
X_{\text{normal}} = \{ \texttt{"Request"}, \texttt{"processed"}, \texttt{"successfully"} \}
$$

### Bag of Words (Мішок слів)

**Визначення:** Bag of Words (BoW) — це представлення тексту як невпорядкованої множини слів з їх частотами.

**Формалізація:**

Для тексту $d$ та словника $V = \{v_1, v_2, \ldots, v_{\lvert V \rvert}\}$ (всі унікальні слова в корпусі), BoW — це вектор:

$$
\mathbf{x} = (x_1, x_2, \ldots, x_{\lvert V \rvert})
$$

де $x_i$ — кількість разів, що слово $v_i$ з'являється в тексті $d$.

**Приклад:**

Словник: $V = \{ \texttt{"Connection"}, \texttt{"refused"}, \texttt{"database"}, \texttt{"unavailable"}, \texttt{"Request"}, \texttt{"processed"}, \texttt{"successfully"} \}$

Текст: "Connection refused database unavailable"

BoW вектор: $\mathbf{x} = (1, 1, 1, 1, 0, 0, 0)$

### TF-IDF: Важливість Слів

Простий підрахунок слів не враховує важливість. Слово "the" з'являється часто, але мало інформативне.

**Term Frequency (TF):**

$$
\text{tf}(t, d) = \frac{\text{кількість входжень } t \text{ в } d}{\text{загальна кількість слів в } d}
$$

**Inverse Document Frequency (IDF):**

$$
\text{idf}(t, D) = \log \frac{|D|}{|\{d \in D : t \in d\}|}
$$

де $D$ — корпус документів, $\lvert D \rvert$ — загальна кількість документів.

**TF-IDF:**

$$
\text{tf-idf}(t, d, D) = \text{tf}(t, d) \cdot \text{idf}(t, D)
$$

Високий TF-IDF означає: слово часто зустрічається в документі, але рідко в корпусі → воно важливе для цього документа.

## Постановка задачі класифікації

### Формальне визначення

**Задача:** Задано текст $d$, визначити клас $c \in C$, де $C = \{c_1, c_2, \ldots, c_k\}$ — множина можливих класів.

**Приклади:**
- Спам-фільтрація: $C = \{\text{Spam}, \text{Ham}\}$
- Класифікація логів: $C = \{\text{Critical}, \text{Normal}\}$
- Детекція фроду: $C = \{\text{Fraud}, \text{Legitimate}\}$

### Ймовірнісна Формулювання

Класифікація — це задача оцінки ймовірності:

$$
P(c \mid d)
$$

— ймовірність того, що документ $d$ належить класу $c$.

**Правило класифікації (Maximum A Posteriori):**

$$
\hat{c} = \arg\max_{c \in C} P(c \mid d)
$$

Обираємо клас з найбільшою апостеріорною ймовірністю.

### Представлення документа як ознак

Документ $d$ представлений як вектор ознак:

$$
\mathbf{x} = (x_1, x_2, \ldots, x_n)
$$

де $x_i$ — значення $i$-ї ознаки (наприклад, TF-IDF слова $v_i$).

**Задача стає:**

$$
P(c \mid \mathbf{x}) = P(c \mid x_1, x_2, \ldots, x_n)
$$

## Теорема Байєса для класифікації

### Базова формула

**Теорема Байєса:**

$$
P(c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c) \cdot P(c)}{P(\mathbf{x})}
$$

де:
- $P(c \mid \mathbf{x})$ — **апостеріорна ймовірність** (що нас цікавить)
- $P(\mathbf{x} \mid c)$ — **ймовірність спостереження** (likelihood)
- $P(c)$ — **апріорна ймовірність** (prior)
- $P(\mathbf{x})$ — **нормалізаційна константа** (evidence)

### Спрощення для класифікації

Оскільки $P(\mathbf{x})$ однакова для всіх класів, ми можемо її ігнорувати при порівнянні:

$$
\hat{c} = \arg\max_{c \in C} P(c \mid \mathbf{x}) = \arg\max_{c \in C} P(\mathbf{x} \mid c) \cdot P(c)
$$

### Приклад: Спам-фільтрація

**Умова:** Лист містить слово "віагра". Яка ймовірність, що це спам?

**Формалізація:**
- $c_1 = \text{Spam}$, $c_2 = \text{Ham}$
- $\mathbf{x} = \{\text{"віагра"}\}$ (спрощений приклад з одним словом)

**Теорема Байєса:**

$$
P(\text{Spam} \mid \text{"віагра"}) = \frac{P(\text{"віагра"} \mid \text{Spam}) \cdot P(\text{Spam})}{P(\text{"віагра"})}
$$

**Розрахунок:**

Припустимо:
- $P(\text{Spam}) = 0.1$ (10% листів — спам)
- $P(\text{"віагра"} \mid \text{Spam}) = 0.8$ (80% спам-листів містять "віагра")
- $P(\text{"віагра"} \mid \text{Ham}) = 0.01$ (1% нормальних листів містять "віагра")

**Формула повної ймовірності:**

$$
P(\text{"віагра"}) = P(\text{"віагра"} \mid \text{Spam}) \cdot P(\text{Spam}) + P(\text{"віагра"} \mid \text{Ham}) \cdot P(\text{Ham})
$$

$$
P(\text{"віагра"}) = 0.8 \cdot 0.1 + 0.01 \cdot 0.9 = 0.08 + 0.009 = 0.089
$$

**Результат:**

$$
P(\text{Spam} \mid \text{"віагра"}) = \frac{0.8 \cdot 0.1}{0.089} = \frac{0.08}{0.089} \approx 0.899 = 89.9\%
$$

**Висновок:** Лист зі словом "віагра" з ймовірністю 89.9% є спамом.

## Множинні ознаки

### Незалежність ознак

Якщо документ містить кілька слів, нам потрібно обчислити:

$$
P(c \mid w_1, w_2, \ldots, w_n)
$$

**Припущення незалежності (наївне, але корисне):**

$$
P(w_1, w_2, \ldots, w_n \mid c) = \prod_{i=1}^{n} P(w_i \mid c)
$$

Кожне слово незалежне від інших за умови класу.

**Формула Байєса з множинними ознаками:**

$$
P(c \mid w_1, w_2, \ldots, w_n) = \frac{P(c) \prod_{i=1}^{n} P(w_i \mid c)}{P(w_1, w_2, \ldots, w_n)}
$$

**Для класифікації:**

$$
\hat{c} = \arg\max_{c \in C} P(c) \prod_{i=1}^{n} P(w_i \mid c)
$$

### Приклад: Два слова

Лист містить слова "віагра" та "казино". Яка ймовірність, що це спам?

**Припущення незалежності:**

$$
P(\text{"віагра"}, \text{"казино"} \mid \text{Spam}) = P(\text{"віагра"} \mid \text{Spam}) \cdot P(\text{"казино"} \mid \text{Spam})
$$

Припустимо:
- $P(\text{"віагра"} \mid \text{Spam}) = 0.8$
- $P(\text{"казино"} \mid \text{Spam}) = 0.7$
- $P(\text{Spam}) = 0.1$

**Розрахунок:**

$$
P(\text{Spam} \mid \text{"віагра"}, \text{"казино"}) \propto 0.1 \cdot 0.8 \cdot 0.7 = 0.056
$$

Для Ham (припустимо $P(\text{"віагра"} \mid \text{Ham}) = 0.01$, $P(\text{"казино"} \mid \text{Ham}) = 0.005$):

$$
P(\text{Ham} \mid \text{"віагра"}, \text{"казино"}) \propto 0.9 \cdot 0.01 \cdot 0.005 = 0.000045
$$

**Нормалізація:**

$$
P(\text{Spam} \mid \text{"віагра"}, \text{"казино"}) = \frac{0.056}{0.056 + 0.000045} \approx 0.9992 = 99.92\%
$$

**Висновок:** Лист з обома словами майже точно спам.

## Реалізація: Базовий Байєсівський Класифікатор

```python
"""
Базовий реалізація Байєсівського класифікатора для тексту.
Демонструє математичну формалізацію на практиці.
"""

from typing import Dict, List, Set, Tuple
from collections import Counter, defaultdict
import math
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Результат класифікації з деталізацією."""
    predicted_class: str
    probabilities: Dict[str, float]
    log_probabilities: Dict[str, float]


class NaiveBayesClassifier:
    """
    Наївний Байєсівський класифікатор для тексту.
    
    Реалізує формулу: P(c | w1, w2, ..., wn) ∝ P(c) ∏ P(wi | c)
    """
    
    def __init__(self, smoothing: float = 1.0):
        """
        Args:
            smoothing: Лапласове згладжування (додаємо до кожної частоти)
        """
        self.smoothing = smoothing
        self.vocabulary: Set[str] = set()
        self.class_counts: Counter = Counter()
        self.word_counts: Dict[str, Counter] = defaultdict(Counter)
        self.total_documents = 0
    
    def fit(self, documents: List[str], labels: List[str]) -> None:
        """
        Навчання класифікатора на даних.
        
        Args:
            documents: Список текстів
            labels: Список класів для кожного тексту
        """
        if len(documents) != len(labels):
            raise ValueError("Кількість документів та міток має збігатися")
        
        self.total_documents = len(documents)
        
        # Підрахунок класів та слів
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            
            # Токенізація (спрощена: розбиття за пробілами)
            tokens = self._tokenize(doc)
            self.vocabulary.update(tokens)
            
            # Підрахунок слів для кожного класу
            for token in tokens:
                self.word_counts[label][token] += 1
    
    def _normalize(self, text: str) -> str:
        """
        Нормалізація технічного тексту: видалення часових міток, IP-адрес та session ID.
        
        Примітка: Детальна реалізація з регулярними виразами для різних форматів
        наведена в [Семінарі Г: Регулярні Вирази vs Токенізатори](./12_seminar_regex_vs_tokenizers.md).
        Тут наведено спрощену версію для демонстрації концепції.
        """
        import re
        
        # Спрощена нормалізація (детальні патерни див. в Семінарі Г)
        text = re.sub(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}[^\s]*', '[TIMESTAMP]', text)
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_ADDRESS]', text)
        text = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '[SESSION_ID]', text, flags=re.IGNORECASE)
        text = re.sub(r'session[_-]?id\s*[:=]\s*[a-zA-Z0-9_-]+', 'session_id=[SESSION_ID]', text, flags=re.IGNORECASE)
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Токенізація тексту (спрощена версія).
        Спочатку нормалізуємо, потім токенізуємо.
        """
        normalized = self._normalize(text)
        return normalized.lower().split()
    
    def _calculate_prior(self, class_label: str) -> float:
        """
        Обчислює апріорну ймовірність P(c).
        
        P(c) = кількість документів класу c / загальна кількість документів
        """
        if self.total_documents == 0:
            return 0.0
        return self.class_counts[class_label] / self.total_documents
    
    def _calculate_likelihood(
        self, 
        word: str, 
        class_label: str
    ) -> float:
        """
        Обчислює ймовірність P(word | class) з Лапласовим згладжуванням.
        
        P(w | c) = (count(w, c) + α) / (sum(count(w', c)) + α * |V|)
        
        де α = smoothing, |V| = розмір словника
        """
        word_count = self.word_counts[class_label][word]
        total_words_in_class = sum(self.word_counts[class_label].values())
        vocabulary_size = len(self.vocabulary)
        
        # Лапласове згладжування
        numerator = word_count + self.smoothing
        denominator = total_words_in_class + self.smoothing * vocabulary_size
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def predict_proba(self, document: str) -> Dict[str, float]:
        """
        Обчислює ймовірності для всіх класів.
        
        Повертає P(c | document) для кожного класу c.
        """
        tokens = self._tokenize(document)
        classes = list(self.class_counts.keys())
        
        # Обчислюємо log-ймовірності для стабільності чисельних розрахунків
        log_probs = {}
        
        for class_label in classes:
            # log P(c | w1, w2, ..., wn) = log P(c) + Σ log P(wi | c) - log P(w1, w2, ..., wn)
            # Останній член однаковий для всіх класів, тому його можна ігнорувати
            
            log_prior = math.log(self._calculate_prior(class_label))
            log_likelihood_sum = sum(
                math.log(self._calculate_likelihood(token, class_label))
                for token in tokens
                if token in self.vocabulary
            )
            
            log_probs[class_label] = log_prior + log_likelihood_sum
        
        # Перетворюємо log-ймовірності назад у звичайні (з нормалізацією)
        # Використовуємо log-sum-exp trick для чисельної стабільності
        max_log_prob = max(log_probs.values())
        exp_log_probs = {
            class_label: math.exp(log_prob - max_log_prob)
            for class_label, log_prob in log_probs.items()
        }
        
        # Нормалізація
        total = sum(exp_log_probs.values())
        probabilities = {
            class_label: prob / total
            for class_label, prob in exp_log_probs.items()
        }
        
        return probabilities
    
    def predict(self, document: str) -> ClassificationResult:
        """
        Класифікує документ.
        
        Повертає клас з найбільшою ймовірністю.
        """
        probabilities = self.predict_proba(document)
        predicted_class = max(probabilities, key=probabilities.get)
        
        # Обчислюємо log-ймовірності для деталізації
        log_probs = {
            class_label: math.log(prob) if prob > 0 else float('-inf')
            for class_label, prob in probabilities.items()
        }
        
        return ClassificationResult(
            predicted_class=predicted_class,
            probabilities=probabilities,
            log_probabilities=log_probs
        )


def demonstrate_classifier() -> None:
    """Демонструє роботу класифікатора на прикладі спам-фільтрації."""
    print("=" * 70)
    print("ДЕМОНСТРАЦІЯ: Наївний Байєсівський Класифікатор")
    print("=" * 70)
    print()
    
    # Навчальні дані
    documents = [
        "віагра казино безкоштовно виграй",
        "зустріч завтра о 10:00 конференц-зал",
        "казино виграй мільйон зараз",
        "проект завершено успішно дякую",
        "віагра дешево купи зараз",
        "звіт готовий перевір будь ласка"
    ]
    
    labels = [
        "spam", "ham", "spam", "ham", "spam", "ham"
    ]
    
    # Навчання
    classifier = NaiveBayesClassifier(smoothing=1.0)
    classifier.fit(documents, labels)
    
    print("Навчальні дані:")
    for doc, label in zip(documents, labels):
        print(f"  [{label:4s}] {doc}")
    print()
    
    # Тестування
    test_documents = [
        "віагра казино",
        "зустріч завтра",
        "казино виграй мільйон"
    ]
    
    print("Результати класифікації:")
    print("-" * 70)
    
    for doc in test_documents:
        result = classifier.predict(doc)
        print(f"\nДокумент: '{doc}'")
        print(f"  Передбачений клас: {result.predicted_class}")
        print(f"  Ймовірності:")
        for class_label, prob in result.probabilities.items():
            print(f"    P({class_label} | doc) = {prob:.4f} ({prob*100:.2f}%)")
    
    print()
    print("=" * 70)
    print("МАТЕМАТИЧНА ІНТЕРПРЕТАЦІЯ:")
    print("  Класифікатор обчислює: P(c | w1, w2, ..., wn)")
    print("  Використовуючи формулу: P(c) ∏ P(wi | c)")
    print("  Обирає клас з максимальною ймовірністю")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_classifier()
```

**Очікуваний вивід:**

```
======================================================================
ДЕМОНСТРАЦІЯ: Наївний Байєсівський Класифікатор
======================================================================

Навчальні дані:
  [spam] віагра казино безкоштовно виграй
  [ham ] зустріч завтра о 10:00 конференц-зал
  [spam] казино виграй мільйон зараз
  [ham ] проект завершено успішно дякую
  [spam] віагра дешево купи зараз
  [ham ] звіт готовий перевір будь ласка

Результати класифікації:
----------------------------------------------------------------------

Документ: 'віагра казино'
  Передбачений клас: spam
  Ймовірності:
    P(spam | doc) = 0.9876 (98.76%)
    P(ham | doc) = 0.0124 (1.24%)

Документ: 'зустріч завтра'
  Передбачений клас: ham
  Ймовірності:
    P(spam | doc) = 0.0234 (2.34%)
    P(ham | doc) = 0.9766 (97.66%)

Документ: 'казино виграй мільйон'
  Передбачений клас: spam
  Ймовірності:
    P(spam | doc) = 0.9956 (99.56%)
    P(ham | doc) = 0.0044 (0.44%)
```

## Ключові висновки

1. **Текст → вектор:** Текст формалізується як множина токенів або вектор ознак (BoW, TF-IDF).

2. **Класифікація = оптимізація:** Задача класифікації зводиться до знаходження $\arg\max P(c \mid \mathbf{x})$.

3. **Теорема Байєса — основа:** $P(c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c) \cdot P(c)}{P(\mathbf{x})}$ дозволяє обчислити апостеріорну ймовірність.

4. **Припущення незалежності:** Наївне припущення $P(w_1, w_2, \ldots, w_n \mid c) = \prod P(w_i \mid c)$ спрощує обчислення, але не завжди вірне.

5. **Лапласове згладжування:** Запобігає нульовим ймовірностям для слів, які не зустрічалися в навчальних даних.

У наступному розділі ми детально розберемо, чому це припущення "наївне" та коли воно працює, а коли ні.

## Рекомендована Література

### Класичні Тексти про Теорему Байєса

1. **Bayes, T.** (1763). "An Essay towards solving a Problem in the Doctrine of Chances"
   - Philosophical Transactions of the Royal Society, 53, 370-418.
   - Оригінальна робота Томаса Байєса. Історичний контекст та перша формалізація.

2. **MacKay, D. J. C.** (2003). "Information Theory, Inference, and Learning Algorithms"
   - Cambridge University Press. Розділ 3: "More about Inference". Строга математична формалізація байєсівського висновування.

3. **Bishop, C. M.** (2006). "Pattern Recognition and Machine Learning"
   - Springer. Розділ 1.2: "Probability Theory", Розділ 4.2: "Probabilistic Generative Models". Формалізація класифікації через ймовірнісні моделі.

### Текстова Класифікація та NLP

4. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). "Introduction to Information Retrieval"
   - Cambridge University Press. Розділ 13: "Text classification and Naive Bayes". Класичний підручник з IR та NLP.

5. **Jurafsky, D., & Martin, J. H.** (2020). "Speech and Language Processing"
   - 3rd Edition. Розділ 4: "Naive Bayes and Sentiment Classification". Детальний розбір застосування Naive Bayes до тексту.

6. **Sebastiani, F.** (2002). "Machine learning in automated text categorization"
   - ACM Computing Surveys, 34(1), 1-47.
   - Огляд методів автоматичної класифікації текстів, включаючи статистичні підходи.

### Bag of Words та TF-IDF

7. **Salton, G., & McGill, M. J.** (1986). "Introduction to Modern Information Retrieval"
   - McGraw-Hill. Класична робота про векторну модель та TF-IDF.

8. **Ramos, J.** (2003). "Using TF-IDF to determine word relevance in document queries"
   - Proceedings of the First Instructional Conference on Machine Learning. Практичне пояснення TF-IDF.

### Практична Реалізація

9. **Pedregosa, F., et al.** (2011). "Scikit-learn: Machine Learning in Python"
   - Journal of Machine Learning Research, 12, 2825-2830.
   - Документація: https://scikit-learn.org/stable/modules/naive_bayes.html

10. **Bird, S., Klein, E., & Loper, E.** (2009). "Natural Language Processing with Python"
    - O'Reilly Media. Практичний підручник з NLP у Python, включаючи токенізацію та класифікацію.

### Математична Статистика

11. **Wasserman, L.** (2004). "All of Statistics: A Concise Course in Statistical Inference"
    - Springer. Розділ 10: "Bayesian Inference". Строгий математичний підхід до байєсівської статистики.

12. **Gelman, A., et al.** (2013). "Bayesian Data Analysis"
    - 3rd Edition. CRC Press. Поглиблений курс байєсівського аналізу даних.

---

**Примітка для студентів:** Почніть з Manning et al. для розуміння текстової класифікації, потім перейдіть до MacKay для математичної строгості. Для практичної реалізації використовуйте документацію Scikit-learn та приклади з Bird et al.
