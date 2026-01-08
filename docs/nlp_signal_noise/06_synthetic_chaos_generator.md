---
title: "Генерація Синтетичного Хаосу: Створення Реалістичних Логів"
layout: default
author: Віталій Платонов
---

# Генерація Синтетичного Хаосу: Створення Реалістичних Логів

Реальні логи містять конфіденційну інформацію: IP-адреси, паролі, токени, персональні дані. Їх не можна використовувати для навчання моделей без ретельного анонімування.

Але як створити синтетичні дані, які зберігають статистичні властивості реальних логів? Як згенерувати "правильний шум" та "рідкісні аномалії" для тестування алгоритмів?

## Проблема Конфіденційності

### Чому Реальні Логи Недоступні

**Конфіденційна інформація в логах:**

1. **Персональні дані:**
   ```
   User login: john.doe@company.com
   Payment processed: card ending 1234
   ```

2. **Безпекові дані:**
   ```
   API key: sk_live_abc123xyz
   Session token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

3. **Інфраструктурні деталі:**
   ```
   Database connection: postgresql://prod-db.internal:5432
   Internal IP: 10.0.0.42
   ```

**Наслідки:**
- GDPR/CCPA вимагають анонімізації
- Анонімізація може зруйнувати структуру даних
- Часткова анонімізація небезпечна (re-identification attacks)

### Альтернатива: Синтетичні Дані

**Переваги:**
- Немає конфіденційної інформації
- Контрольований розподіл класів
- Відтворюваність (reproducibility)
- Можливість тестувати edge cases

**Вимоги:**
- Реалістична структура
- Правильний "шум" (нормальні події)
- Рідкісні аномалії з правильним base rate
- Різноманітність формулювань

## Математична Формалізація

### Розподіл Класів

**Незбалансований датасет:**

$$P(\text{Normal}) = 0.99$$
$$P(\text{Critical}) = 0.01$$

**Для $N$ записів:**

$$N_{\text{Normal}} = \lfloor 0.99 \times N \rfloor$$
$$N_{\text{Critical}} = N - N_{\text{Normal}}$$

### Генерація Текстів

**Нормальні логи:** Випадкові комбінації шаблонів:

$$\text{Log}_{\text{Normal}} \sim \text{Template}(\text{verbs}, \text{nouns}, \text{adjectives})$$

**Критичні логи:** Структуровані шаблони з помилками:

$$\text{Log}_{\text{Critical}} \sim \text{ErrorTemplate}(\text{error\_type}, \text{component})$$

### Варіативність через Шум

**Додавання варіативності:**

$$\text{Log}_{\text{final}} = \text{Log}_{\text{template}} + \text{Noise}(\sigma)$$

де $\text{Noise}(\sigma)$ — випадкові варіації (синоніми, порядок слів, додаткові деталі).

## Генерація через Шаблони

### Структура Шаблонів

**Нормальні логи:**

```python
normal_templates = [
    "Request {verb} {status}",
    "{component} {action} {status}",
    "{service} {state} {details}"
]
```

**Критичні логи:**

```python
critical_templates = [
    "{component} {error_verb} {error_type}",
    "{error_type}: {component} {error_details}",
    "{component} {error_state}: {error_reason}"
]
```

### Заповнення Шаблонів

**Словники:**

- `verbs`: ["processed", "completed", "executed", "handled"]
- `error_verbs`: ["failed", "refused", "timeout", "crashed"]
- `components`: ["database", "API", "service", "cache"]
- `error_types`: ["connection", "timeout", "memory", "disk"]

**Вибір:** Випадковий вибір з розподілу (можна зважений).

## Реалізація: Генератор Синтетичних Логів

```python
"""
Генератор синтетичних технічних логів.
Створює незбалансований датасет (99% Normal, 1% Critical)
для тестування алгоритмів класифікації.
"""

from typing import List, Tuple, Dict
import random
from dataclasses import dataclass
from faker import Faker
import numpy as np


@dataclass
class LogEntry:
    """Окремий запис логу."""
    message: str
    label: str
    timestamp: str = ""
    metadata: Dict = None


class SyntheticLogGenerator:
    """
    Генератор синтетичних технічних логів.
    
    Створює реалістичні логи з правильним розподілом класів
    та варіативністю формулювань.
    """
    
    def __init__(self, seed: int = 42):
        """
        Args:
            seed: Seed для відтворюваності
        """
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Шаблони для нормальних логів
        self.normal_templates = [
            "Request {verb} {status}",
            "{component} {action} {status}",
            "{service} {state} successfully",
            "User {user_action} {component}",
            "{component} {verb} in {time}ms",
            "{service} {state} {details}",
            "Cache {cache_action} for {key}",
            "Database query {query_status}",
            "API endpoint {endpoint} {verb} {status}",
            "Session {session_action} for user {user_id}"
        ]
        
        # Шаблони для критичних логів
        self.critical_templates = [
            "{component} {error_verb} {error_type}",
            "{error_type}: {component} {error_details}",
            "{component} {error_state}: {error_reason}",
            "ERROR: {component} {error_verb} after {timeout}s",
            "CRITICAL: {component} {error_state} - {error_reason}",
            "{component} connection {error_verb} - {error_details}",
            "FATAL: {component} {error_type} detected",
            "{component} {error_verb} {error_type} - {error_reason}",
            "PANIC: {component} {error_state}",
            "{error_type} in {component}: {error_details}"
        ]
        
        # Словники для заповнення шаблонів
        self.verbs = [
            "processed", "completed", "executed", "handled",
            "served", "delivered", "responded", "acknowledged"
        ]
        
        self.error_verbs = [
            "failed", "refused", "timeout", "crashed",
            "aborted", "rejected", "terminated", "lost"
        ]
        
        self.components = [
            "database", "API", "service", "cache",
            "queue", "storage", "network", "auth"
        ]
        
        self.error_types = [
            "connection", "timeout", "memory", "disk",
            "network", "authentication", "authorization", "resource"
        ]
        
        self.status_words = [
            "successfully", "completed", "ok", "ready",
            "available", "online", "active", "healthy"
        ]
        
        self.error_states = [
            "unavailable", "down", "offline", "unreachable",
            "overloaded", "exhausted", "corrupted", "broken"
        ]
    
    def _fill_template(
        self, 
        template: str, 
        is_critical: bool = False
    ) -> str:
        """
        Заповнює шаблон випадковими значеннями.
        
        Args:
            template: Шаблон з плейсхолдерами
            is_critical: Чи це критичний лог
        """
        replacements = {}
        
        if is_critical:
            replacements.update({
                "verb": random.choice(self.error_verbs),
                "error_verb": random.choice(self.error_verbs),
                "component": random.choice(self.components),
                "error_type": random.choice(self.error_types),
                "error_state": random.choice(self.error_states),
                "error_details": self._generate_error_details(),
                "error_reason": self._generate_error_reason(),
                "timeout": random.randint(5, 60),
                "time": random.randint(10, 1000)
            })
        else:
            replacements.update({
                "verb": random.choice(self.verbs),
                "action": random.choice(self.verbs),
                "component": random.choice(self.components),
                "service": random.choice(self.components),
                "state": random.choice(self.status_words),
                "status": random.choice(self.status_words),
                "details": self._generate_normal_details(),
                "user_action": random.choice(["accessed", "logged into", "queried"]),
                "cache_action": random.choice(["hit", "miss", "updated"]),
                "query_status": random.choice(["executed", "completed", "returned"]),
                "endpoint": f"/api/v{random.randint(1,3)}/{random.choice(['users', 'data', 'status'])}",
                "session_action": random.choice(["created", "validated", "expired"]),
                "key": f"key_{random.randint(1000, 9999)}",
                "user_id": f"user_{random.randint(100, 999)}",
                "time": random.randint(1, 100)
            })
        
        # Заповнюємо шаблон
        result = template
        for key, value in replacements.items():
            result = result.replace(f"{{{key}}}", str(value))
        
        return result
    
    def _generate_error_details(self) -> str:
        """Генерує деталі помилки."""
        details = [
            "connection refused",
            "timeout after 30s",
            "out of memory",
            "disk full",
            "network partition",
            "authentication failed",
            "resource exhausted",
            "corrupted data"
        ]
        return random.choice(details)
    
    def _generate_error_reason(self) -> str:
        """Генерує причину помилки."""
        reasons = [
            "insufficient resources",
            "network failure",
            "configuration error",
            "data corruption",
            "service unavailable",
            "authentication timeout",
            "disk quota exceeded",
            "memory limit reached"
        ]
        return random.choice(reasons)
    
    def _generate_normal_details(self) -> str:
        """Генерує деталі для нормальних логів."""
        details = [
            "all systems operational",
            "response time optimal",
            "cache performance good",
            "no issues detected",
            "health check passed",
            "metrics within range"
        ]
        return random.choice(details)
    
    def generate_normal_log(self) -> LogEntry:
        """Генерує один нормальний лог."""
        template = random.choice(self.normal_templates)
        message = self._fill_template(template, is_critical=False)
        
        # Додаємо варіативність: іноді додаємо додаткові деталі
        if random.random() < 0.3:
            message += f" (duration: {random.randint(1, 100)}ms)"
        
        return LogEntry(
            message=message,
            label="normal",
            timestamp=self.faker.iso8601(),
            metadata={"source": "synthetic"}
        )
    
    def generate_critical_log(self) -> LogEntry:
        """Генерує один критичний лог."""
        template = random.choice(self.critical_templates)
        message = self._fill_template(template, is_critical=True)
        
        return LogEntry(
            message=message,
            label="critical",
            timestamp=self.faker.iso8601(),
            metadata={"source": "synthetic", "severity": "high"}
        )
    
    def generate_dataset(
        self, 
        total_size: int,
        critical_rate: float = 0.01
    ) -> List[LogEntry]:
        """
        Генерує незбалансований датасет.
        
        Args:
            total_size: Загальна кількість записів
            critical_rate: Частота критичних логів (0.0 - 1.0)
        
        Returns:
            Список LogEntry з правильним розподілом класів
        """
        n_critical = int(total_size * critical_rate)
        n_normal = total_size - n_critical
        
        logs = []
        
        # Генеруємо критичні логи
        for _ in range(n_critical):
            logs.append(self.generate_critical_log())
        
        # Генеруємо нормальні логи
        for _ in range(n_normal):
            logs.append(self.generate_normal_log())
        
        # Перемішуємо для реалістичності
        random.shuffle(logs)
        
        return logs
    
    def add_variations(
        self, 
        logs: List[LogEntry],
        variation_rate: float = 0.1
    ) -> List[LogEntry]:
        """
        Додає варіації до логів (синоніми, перестановки).
        
        Args:
            logs: Список логів
            variation_rate: Частота застосування варіацій
        """
        variations = []
        
        for log in logs:
            if random.random() < variation_rate:
                # Додаємо варіацію: замінюємо слова на синоніми
                message = self._apply_synonym_variation(log.message)
                variations.append(LogEntry(
                    message=message,
                    label=log.label,
                    timestamp=log.timestamp,
                    metadata=log.metadata
                ))
            else:
                variations.append(log)
        
        return variations
    
    def _apply_synonym_variation(self, message: str) -> str:
        """Застосовує синонімічні заміни."""
        synonyms = {
            "database": ["DB", "data store", "repository"],
            "failed": ["crashed", "aborted", "terminated"],
            "connection": ["conn", "link", "session"],
            "timeout": ["expired", "exceeded limit"],
            "successfully": ["ok", "completed", "done"]
        }
        
        result = message
        for word, syns in synonyms.items():
            if word.lower() in result.lower() and random.random() < 0.3:
                result = result.replace(word, random.choice(syns), 1)
        
        return result


def demonstrate_generator() -> None:
    """Демонструє роботу генератора синтетичних логів."""
    print("=" * 70)
    print("ГЕНЕРАТОР СИНТЕТИЧНИХ ЛОГІВ")
    print("=" * 70)
    print()
    
    generator = SyntheticLogGenerator(seed=42)
    
    # Генеруємо невеликий датасет для демонстрації
    print("Генерація датасету (1000 записів, 1% критичних)...")
    logs = generator.generate_dataset(total_size=1000, critical_rate=0.01)
    
    # Статистика
    normal_count = sum(1 for log in logs if log.label == "normal")
    critical_count = sum(1 for log in logs if log.label == "critical")
    
    print(f"\nСтатистика:")
    print(f"  Всього записів: {len(logs)}")
    print(f"  Нормальних: {normal_count} ({normal_count/len(logs)*100:.2f}%)")
    print(f"  Критичних: {critical_count} ({critical_count/len(logs)*100:.2f}%)")
    print()
    
    # Приклади нормальних логів
    print("Приклади нормальних логів:")
    print("-" * 70)
    normal_examples = [log for log in logs if log.label == "normal"][:5]
    for i, log in enumerate(normal_examples, 1):
        print(f"  {i}. {log.message}")
    print()
    
    # Приклади критичних логів
    print("Приклади критичних логів:")
    print("-" * 70)
    critical_examples = [log for log in logs if log.label == "critical"][:5]
    for i, log in enumerate(critical_examples, 1):
        print(f"  {i}. {log.message}")
    print()
    
    # Перевірка Пастки Байєса
    print("Перевірка Пастки Байєса:")
    print("-" * 70)
    print("  Якщо класифікатор має точність 99%,")
    print("  але base rate критичних = 1%,")
    print("  то Precision позитивних результатів буде низькою.")
    print()
    print("  Цей датасет дозволяє протестувати це на практиці.")
    print()


def demonstrate_variations() -> None:
    """Демонструє додавання варіацій."""
    print("\n" + "=" * 70)
    print("ВАРІАЦІЇ ТА СИНОНІМИ")
    print("=" * 70)
    print()
    
    generator = SyntheticLogGenerator(seed=42)
    
    # Генеруємо базовий лог
    base_log = generator.generate_critical_log()
    print(f"Базовий лог: {base_log.message}")
    print()
    
    # Додаємо варіації
    variations = []
    for _ in range(5):
        var_log = generator.generate_critical_log()
        variations.append(var_log.message)
    
    print("Варіації (різні формулювання тієї ж проблеми):")
    for i, var in enumerate(variations, 1):
        print(f"  {i}. {var}")
    print()
    
    print("Це демонструє, як одна проблема може бути описана")
    print("різними словами, що є викликом для Bag of Words,")
    print("але не для BERT з контекстними embeddings.")


if __name__ == "__main__":
    try:
        demonstrate_generator()
        demonstrate_variations()
    except ImportError:
        print("Помилка: Потрібно встановити Faker")
        print("  pip install Faker")
```

## Генерація через LLM (Альтернативний Підхід)

### Використання Великих Мовних Моделей

**Ідея:** Використовувати LLM (GPT, Claude) для генерації реалістичних логів через промпти.

**Переваги:**
- Більша варіативність
- Більш реалістичні формулювання
- Менше ручної роботи

**Недоліки:**
- Залежність від API
- Витрати на токени
- Менша контрольованість

### Приклад Промпту

```
Generate 100 realistic technical log entries for a web application.
Requirements:
- 99 should be normal (successful operations)
- 1 should be critical (errors, failures)
- Include variety: database, API, cache, network
- Use realistic error messages
- Format: one log per line
```

## Валідація Синтетичних Даних

### Перевірка Розподілу

**Очікуваний розподіл:**

$$P(\text{Normal}) = 0.99 \pm \epsilon$$
$$P(\text{Critical}) = 0.01 \pm \epsilon$$

де $\epsilon$ — допустима похибка (наприклад, 0.001).

### Перевірка Варіативності

**Метрики:**

1. **Унікальність:** Відсоток унікальних повідомлень
2. **Розподіл довжини:** Середня довжина повідомлень
3. **Розподіл слів:** Частота ключових слів

### Перевірка Реалістичності

**Критерії:**

- Логи виглядають як реальні
- Немає очевидних патернів
- Варіативність формулювань
- Правильна структура (timestamp, level, message)

## Ключові Висновки

1. **Синтетичні дані безпечні:** Немає конфіденційної інформації, можна вільно використовувати.

2. **Контрольований розподіл:** Можна створити точний base rate для тестування Пастки Байєса.

3. **Варіативність важлива:** Різні формулювання однієї проблеми тестують стійкість моделей.

4. **Шаблони + Шум:** Комбінація структурованих шаблонів та випадкових варіацій створює реалістичні дані.

5. **Валідація критична:** Потрібно перевіряти, що синтетичні дані зберігають властивості реальних.

У наступному розділі ми використаємо ці синтетичні дані для навчання та порівняння різних моделей класифікації.

## Рекомендована Література

### Синтетичні Дані та Privacy

1. **Dwork, C.** (2006). "Differential Privacy"
   - ICALP. Фундаментальна робота про захист приватності в даних.

2. **Rubin, D. B.** (1993). "Discussion: Statistical Disclosure Limitation"
   - Journal of Official Statistics, 9(2), 461-468. Проблеми анонімізації даних.

### Генерація Текстових Даних

3. **Goodfellow, I., et al.** (2014). "Generative Adversarial Nets"
   - NIPS. GAN для генерації даних (можна застосувати до текстів).

4. **Bowman, S. R., et al.** (2015). "Generating Sentences from a Continuous Space"
   - CoNLL. VAE для генерації речень.

### Faker та Синтетичні Дані

5. **Faker Documentation**
   - URL: https://faker.readthedocs.io/
   - Бібліотека для генерації реалістичних фейкових даних.

6. **Synthetic Data Vault (SDV)**
   - URL: https://sdv.dev/
   - Бібліотека для генерації синтетичних табличних даних.

### LLM для Генерації

7. **Brown, T., et al.** (2020). "Language Models are Few-Shot Learners"
   - NeurIPS. GPT-3 та можливості генерації через промпти.

8. **OpenAI API Documentation**
   - URL: https://platform.openai.com/docs/guides/text-generation
   - Практичний гайд з використанням GPT для генерації.

### Валідація Синтетичних Даних

9. **Patki, N., Wedge, R., & Veeramachaneni, K.** (2016). "The Synthetic Data Vault"
   - IEEE DSAA. Методи валідації синтетичних даних.

10. **Jordon, J., et al.** (2022). "Evaluating the Quality of Synthetic Data"
    - NeurIPS. Метрики для оцінки якості синтетичних даних.

### Практичні Гайди

11. **Chawla, N. V., et al.** (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
    - JAIR. Методи балансування незбалансованих датасетів.

12. **Fernández, A., et al.** (2018). "An insight into imbalanced Big Data classification"
    - Knowledge-Based Systems. Проблеми та рішення для незбалансованих даних.

---

**Примітка для студентів:** Почніть з Faker documentation для практичної генерації, потім перейдіть до Dwork про differential privacy для розуміння проблем конфіденційності. Для валідації використовуйте метрики з Patki et al. та Jordon et al.





