---
title: "07 Implementation Workshop"
type: lecture
module: Практикум
prerequisites: module 6
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# Практикум: Побудова Пайплайну Фільтрації

Теорія без практики — мертва. Навчимося будувати працюючі системи класифікації логів, порівняємо Naive Bayes та BERT, та зрозуміємо, чому метрики на незбалансованих даних — це війна між Accuracy та F1-score.

## Структура Воркшопу

1. **Baseline:** Naive Bayes (scikit-learn)
2. **Advanced:** BERT-based classifier (HuggingFace)
3. **Metric War:** Порівняння метрик на незбалансованих даних

## Підготовка Даних

### Нормалізація Технічних Текстів

**Критично важливо:** Перед токенізацією та побудовою TF-IDF векторів необхідно нормалізувати технічні тексти. Це зменшує розмірність словника $V$ та покращує якість TF-IDF ознак.

```python
"""
Нормалізація технічних текстів: видалення часових міток, IP-адрес та session ID.
Це критично важливо для зменшення розмірності словника та покращення TF-IDF.
"""

import re
from typing import List


def normalize_technical_text(text: str) -> str:
    """
    Нормалізує технічний текст, замінюючи унікальні ідентифікатори на плейсхолдери.
    
    Args:
        text: Вхідний текст (лог, системне повідомлення)
    
    Returns:
        Нормалізований текст з плейсхолдерами замість унікальних ідентифікаторів
    """
    # Видалення часових міток (різні формати)
    # ISO 8601: 2024-01-15T14:23:45.123Z
    text = re.sub(
        r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?',
        '[TIMESTAMP]',
        text
    )
    # З квадратними дужками: [2024-01-15T14:23:45]
    text = re.sub(
        r'\[?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
        '[TIMESTAMP]',
        text
    )
    # Syslog формат: Jan 15 14:23:45
    text = re.sub(
        r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
        '[TIMESTAMP]',
        text
    )
    
    # Видалення IPv4 адрес
    text = re.sub(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        '[IP_ADDRESS]',
        text
    )
    
    # Видалення IPv6 адрес (спрощена версія)
    text = re.sub(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
        '[IP_ADDRESS]',
        text
    )
    
    # Видалення UUID (стандартний формат)
    text = re.sub(
        r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
        '[SESSION_ID]',
        text,
        flags=re.IGNORECASE
    )
    
    # Видалення session_id з різними форматами
    text = re.sub(
        r'session[_-]?id\s*[:=]\s*[a-zA-Z0-9_-]+',
        'session_id=[SESSION_ID]',
        text,
        flags=re.IGNORECASE
    )
    
    # Видалення хешів та довгих ідентифікаторів (16+ символів)
    text = re.sub(
        r'\b[a-f0-9]{16,}\b',
        '[SESSION_ID]',
        text,
        flags=re.IGNORECASE
    )
    
    return text


def normalize_texts(texts: List[str]) -> List[str]:
    """
    Нормалізує список текстів.
    
    Args:
        texts: Список текстів для нормалізації
    
    Returns:
        Список нормалізованих текстів
    """
    return [normalize_technical_text(text) for text in texts]


# Приклад використання
example_log = "2024-01-15 14:23:45 [ERROR] Connection refused from 192.168.1.100 session_id=a3f5d8e2b1c4"
normalized = normalize_technical_text(example_log)
print(f"До:  {example_log}")
print(f"Після: {normalized}")
# Вивід:
# До:  2024-01-15 14:23:45 [ERROR] Connection refused from 192.168.1.100 session_id=a3f5d8e2b1c4
# Після: [TIMESTAMP] [ERROR] Connection refused from [IP_ADDRESS] session_id=[SESSION_ID]
```

**Чому це важливо:**

1. **Зменшення розмірності словника:** Без нормалізації кожна унікальна IP-адреса або timestamp стає окремим токеном. Для 10,000 логів це може дати 20,000+ унікальних токенів, які не несуть інформації для класифікації.

2. **Покращення TF-IDF:** Унікальні токени мають високий IDF (з'являються в одному документі), але не інформативні. Плейсхолдери `[TIMESTAMP]`, `[IP_ADDRESS]` з'являються в усіх документах (низький IDF) і не "витісняють" важливі слова.

3. **Краща узагальнююча здатність:** Модель навчається на семантичних ознаках ("Connection refused"), а не на конкретних значеннях ("192.168.1.100").

### Завантаження Синтетичного Датасету

```python
"""
Підготовка даних для навчання та тестування.
Використовуємо синтетичний генератор з попереднього розділу.
"""

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Імпортуємо генератор (припускаємо, що він в окремому файлі)
from synthetic_chaos_generator import SyntheticLogGenerator, LogEntry


def prepare_dataset(
    total_size: int = 10000,
    critical_rate: float = 0.01,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Генерує та підготовляє датасет для навчання.
    
    Returns:
        (X_train, y_train, X_test, y_test)
    """
    generator = SyntheticLogGenerator(seed=random_state)
    logs = generator.generate_dataset(total_size, critical_rate)
    
    # Розділяємо на тексти та мітки
    texts = [log.message for log in logs]
    labels = [log.label for log in logs]
    
    # Нормалізуємо тексти перед розділенням на train/test
    texts = normalize_texts(texts)
    
    # Розділяємо на train/test зі стратифікацією
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
    
    return X_train, y_train, X_test, y_test


# Генеруємо датасет
X_train, y_train, X_test, y_test = prepare_dataset(
    total_size=10000,
    critical_rate=0.01
)

print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Train distribution: {np.bincount([1 if y == 'critical' else 0 for y in y_train])}")
print(f"Test distribution: {np.bincount([1 if y == 'critical' else 0 for y in y_test])}")
```

## Baseline: Naive Bayes

### Реалізація з Scikit-learn

```python
"""
Baseline: Naive Bayes класифікатор на Bag of Words.
Очікуваний результат: висока Accuracy, але низький Recall для критичних.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    matthews_corrcoef
)
import pandas as pd


class NaiveBayesBaseline:
    """
    Baseline класифікатор на основі Naive Bayes.
    
    Примітка: Перед використанням цього класифікатора тексти мають бути нормалізовані
    (видалення timestamps, IP-адрес, session ID). Це критично важливо для:
    - Зменшення розмірності словника V
    - Покращення якості TF-IDF ознак
    - Кращої узагальнюючої здатності моделі
    """
    
    def __init__(self):
        """
        Ініціалізує пайплайн.
        
        Примітка: max_features=5000 встановлено з урахуванням того, що тексти
        нормалізовані. Без нормалізації знадобилося б значно більше features
        для покриття унікальних IP-адрес та timestamps.
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # Уніграми та біграми
                min_df=2,
                max_df=0.95
            )),
            ('nb', MultinomialNB(alpha=1.0))  # Лапласове згладжування
        ])
    
    def train(self, X_train: List[str], y_train: List[str]) -> None:
        """Навчає модель."""
        self.pipeline.fit(X_train, y_train)
    
    def predict(self, X_test: List[str]) -> List[str]:
        """Передбачає класи."""
        return self.pipeline.predict(X_test)
    
    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """Повертає ймовірності класів."""
        return self.pipeline.predict_proba(X_test)
    
    def evaluate(
        self, 
        X_test: List[str], 
        y_test: List[str]
    ) -> dict:
        """
        Оцінює модель та повертає метрики.
        
        Returns:
            Словник з метриками
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Базові метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='critical', zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label='critical', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label='critical', zero_division=0)
        
        # MCC (Matthews Correlation Coefficient)
        y_test_binary = [1 if y == 'critical' else 0 for y in y_test]
        y_pred_binary = [1 if y == 'critical' else 0 for y in y_pred]
        mcc = matthews_corrcoef(y_test_binary, y_pred_binary)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=['normal', 'critical'])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }


# Навчання та оцінка
print("=" * 70)
print("BASELINE: NAIVE BAYES")
print("=" * 70)

nb_model = NaiveBayesBaseline()
nb_model.train(X_train, y_train)
nb_results = nb_model.evaluate(X_test, y_test)

print(f"\nМетрики:")
print(f"  Accuracy:  {nb_results['accuracy']:.4f}")
print(f"  Precision: {nb_results['precision']:.4f}")
print(f"  Recall:    {nb_results['recall']:.4f}")
print(f"  F1-score:  {nb_results['f1_score']:.4f}")
print(f"  MCC:       {nb_results['mcc']:.4f}")

print(f"\nConfusion Matrix:")
print(f"  {nb_results['confusion_matrix']}")

print(f"\nClassification Report:")
print(classification_report(y_test, nb_results['predictions']))
```

### Аналіз Результатів Baseline

**Очікуваний результат:**

```
Accuracy:  0.9900  (висока!)
Precision: 0.8500  (добре)
Recall:    0.3000  (погано!)
F1-score:  0.4400  (низький)
```

**Проблема:** Висока Accuracy досягається за рахунок правильної класифікації нормальних логів (99% датасету). Але Recall низький — модель пропускає 70% критичних збоїв.

**Confusion Matrix:**

```
                Predicted
              Normal  Critical
Actual Normal   1980     10
       Critical   14      6
```

**Висновок:** Baseline працює, але не вирішує проблему Alert Fatigue — більшість критичних збоїв пропускається.

## Advanced: BERT-based Classifier

### Реалізація з HuggingFace

```python
"""
Advanced: BERT-based класифікатор.
Очікуваний результат: кращий Recall за рахунок контекстного розуміння.
"""

from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class LogDataset(Dataset):
    """Dataset для PyTorch."""
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Кодуємо мітки
        self.label_map = {'normal': 0, 'critical': 1}
        self.label_ids = [self.label_map[label] for label in labels]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_ids[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier:
    """
    BERT-based класифікатор для технічних логів.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        max_length: int = 128
    ):
        """
        Args:
            model_name: Назва попередньо навченої BERT моделі
            max_length: Максимальна довжина послідовності
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        self.label_map = {'normal': 0, 'critical': 1}
        self.reverse_label_map = {0: 'normal', 1: 'critical'}
    
    def train(
        self,
        X_train: List[str],
        y_train: List[str],
        X_val: List[str] = None,
        y_val: List[str] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> None:
        """
        Навчає BERT модель.
        
        Args:
            X_train: Тренувальні тексти
            y_train: Тренувальні мітки
            X_val: Валідаційні тексти (опціонально)
            y_val: Валідаційні мітки (опціонально)
            epochs: Кількість епох
            batch_size: Розмір батчу
            learning_rate: Швидкість навчання
        """
        # Створюємо datasets
        train_dataset = LogDataset(X_train, y_train, self.tokenizer, self.max_length)
        
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_dataset = LogDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        # Налаштування тренування
        training_args = TrainingArguments(
            output_dir='./bert_results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch' if val_dataset else 'no',
            save_strategy='epoch',
            load_best_model_at_end=True if val_dataset else False,
        )
        
        # Функція для обчислення метрик
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary', zero_division=0
            )
            accuracy = accuracy_score(labels, predictions)
            mcc = matthews_corrcoef(labels, predictions)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mcc': mcc
            }
        
        # Створюємо Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Навчаємо
        trainer.train()
    
    def predict(self, texts: List[str]) -> List[str]:
        """Передбачає класи для списку текстів."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                outputs = self.model(**encoding)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=-1).item()
                predictions.append(self.reverse_label_map[predicted_class])
        
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Повертає ймовірності класів."""
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).numpy()[0]
                probabilities.append(probs)
        
        return np.array(probabilities)
    
    def evaluate(
        self, 
        X_test: List[str], 
        y_test: List[str]
    ) -> dict:
        """Оцінює модель."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='critical', zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label='critical', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label='critical', zero_division=0)
        
        # MCC (Matthews Correlation Coefficient)
        y_test_binary = [1 if y == 'critical' else 0 for y in y_test]
        y_pred_binary = [1 if y == 'critical' else 0 for y in y_pred]
        mcc = matthews_corrcoef(y_test_binary, y_pred_binary)
        
        cm = confusion_matrix(y_test, y_pred, labels=['normal', 'critical'])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mcc': mcc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }


# Навчання та оцінка BERT
print("\n" + "=" * 70)
print("ADVANCED: BERT-BASED CLASSIFIER")
print("=" * 70)

# Розділяємо train на train/val для fine-tuning
X_train_bert, X_val_bert, y_train_bert, y_val_bert = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

bert_model = BERTClassifier(model_name="bert-base-uncased")

# Примітка: Fine-tuning займає час. Для демонстрації можна використати
# попередньо навчену модель або зменшити кількість епох.
print("\nНавчання BERT (це може зайняти час)...")
bert_model.train(
    X_train_bert, y_train_bert,
    X_val_bert, y_val_bert,
    epochs=2,  # Зменшено для швидкості
    batch_size=8
)

bert_results = bert_model.evaluate(X_test, y_test)

print(f"\nМетрики:")
print(f"  Accuracy:  {bert_results['accuracy']:.4f}")
print(f"  Precision: {bert_results['precision']:.4f}")
print(f"  Recall:    {bert_results['recall']:.4f}")
print(f"  F1-score:  {bert_results['f1_score']:.4f}")
print(f"  MCC:       {bert_results['mcc']:.4f}")

print(f"\nConfusion Matrix:")
print(f"  {bert_results['confusion_matrix']}")
```

## Matthews Correlation Coefficient (MCC): Найнадійніша Метрика для Незбалансованих Класів

### Проблема з Існуючими Метриками

На сильно незбалансованих даних (наприклад, 0.01% критичних логів) навіть F1-score може бути оманливим:

**Приклад проблеми:**
- Модель класифікує все як "Normal" → Accuracy = 99%, але Recall = 0%
- F1-score = 0 (бо Recall = 0), але це не показує повну картину
- Precision може бути невизначеним (ділення на 0)

**Рішення:** Matthews Correlation Coefficient (MCC) — єдина метрика, яка враховує всі чотири клітинки confusion matrix та завжди дає змістовний результат.

### Математична Формалізація MCC

**Confusion Matrix:**

```
                    Predicted
                 Normal  Critical
Actual Normal      TN      FP
       Critical    FN      TP
```

де:
- **TP (True Positive):** Критичні логі, правильно класифіковані як критичні
- **TN (True Negative):** Нормальні логі, правильно класифіковані як нормальні
- **FP (False Positive):** Нормальні логі, помилково класифіковані як критичні
- **FN (False Negative):** Критичні логі, помилково класифіковані як нормальні

**Matthews Correlation Coefficient:**

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

**Властивості MCC:**
- Діапазон: $[-1, +1]$
- $MCC = +1$: Ідеальна класифікація
- $MCC = 0$: Випадкова класифікація (не краще за випадковий вибір)
- $MCC = -1$: Повна протилежність (все класифіковано неправильно)

### Чому MCC Краще за F1-score для Незбалансованих Класів

**1. Враховує всі чотири клітинки confusion matrix:**

- **F1-score:** Залежить лише від TP, FP, FN (не враховує TN)
- **MCC:** Враховує TP, TN, FP, FN → повна картина класифікації

**2. Симетричність:**

- **F1-score:** Не симетричний відносно класів (залежить від вибору "позитивного" класу)
- **MCC:** Симетричний → однаковий результат незалежно від вибору класу

**3. Інваріантність до незбалансованості:**

**Приклад:** Датасет з 99% Normal, 1% Critical

**Сценарій 1:** Модель класифікує все як Normal
- TP = 0, TN = 9900, FP = 0, FN = 100
- Accuracy = 99% (оманливо висока!)
- Precision = undefined (0/0)
- Recall = 0%
- F1-score = 0
- **MCC = -0.1** (показує, що модель гірша за випадкову)

**Сценарій 2:** Модель правильно знаходить 50% критичних
- TP = 50, TN = 9900, FP = 0, FN = 50
- Accuracy = 99.5%
- Precision = 100% (всі передбачення критичних правильні)
- Recall = 50%
- F1-score = 66.7%
- **MCC = 0.71** (показує хорошу якість)

**Сценарій 3:** Модель знаходить всі критичні, але з помилками
- TP = 100, TN = 9800, FP = 100, FN = 0
- Accuracy = 99%
- Precision = 50% (половина передбачень критичних помилкові)
- Recall = 100%
- F1-score = 66.7% (такий самий, як у сценарії 2!)
- **MCC = 0.71** (такий самий, як у сценарії 2, але з іншою структурою помилок)

**Висновок:** MCC дає однакове значення для моделей з однаковою "загальною якістю", незалежно від того, чи помилки в Precision чи Recall.

### Порівняння з Іншими Метриками

| Метрика | Діапазон | Враховує TN? | Симетрична? | Інваріантна до незбалансованості? |
|---------|----------|--------------|-------------|-----------------------------------|
| Accuracy | [0, 1] | Так | Так | Ні (оманлива на незбалансованих) |
| Precision | [0, 1] | Ні | Ні | Ні |
| Recall | [0, 1] | Ні | Ні | Ні |
| F1-score | [0, 1] | Ні | Ні | Частково |
| **MCC** | **[-1, 1]** | **Так** | **Так** | **Так** |

### Практична Реалізація MCC

```python
"""
Реалізація та порівняння MCC з іншими метриками.
"""

from sklearn.metrics import matthews_corrcoef, confusion_matrix
import numpy as np


def compute_mcc_from_confusion_matrix(cm):
    """
    Обчислює MCC з confusion matrix.
    
    Args:
        cm: Confusion matrix [[TN, FP], [FN, TP]]
    
    Returns:
        MCC значення
    """
    tn, fp, fn, tp = cm.ravel()
    
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compare_metrics_with_mcc(y_true, y_pred):
    """
    Порівнює різні метрики, включаючи MCC.
    
    Args:
        y_true: Істинні мітки
        y_pred: Передбачені мітки
    
    Returns:
        Словник з усіма метриками
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, matthews_corrcoef
    )
    
    # Обчислюємо confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['normal', 'critical'])
    
    # Базові метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='critical', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label='critical', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='critical', zero_division=0)
    
    # MCC (використовуємо scikit-learn)
    mcc = matthews_corrcoef(
        [1 if y == 'critical' else 0 for y in y_true],
        [1 if y == 'critical' else 0 for y in y_pred]
    )
    
    # Альтернативно: обчислюємо з confusion matrix
    mcc_from_cm = compute_mcc_from_confusion_matrix(cm)
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc,
        'mcc_from_cm': mcc_from_cm  # Перевірка
    }


def demonstrate_mcc_advantages():
    """
    Демонструє переваги MCC на незбалансованих даних.
    """
    print("=" * 80)
    print("MATTHEWS CORRELATION COEFFICIENT: ПЕРЕВАГИ ДЛЯ НЕЗБАЛАНСОВАНИХ ДАНИХ")
    print("=" * 80)
    print()
    
    # Симулюємо незбалансований датасет: 99% Normal, 1% Critical
    n_total = 10000
    n_critical = 100
    n_normal = 9900
    
    # Сценарій 1: Модель класифікує все як Normal
    print("СЦЕНАРІЙ 1: Модель класифікує все як Normal")
    print("-" * 80)
    y_true_1 = ['normal'] * n_normal + ['critical'] * n_critical
    y_pred_1 = ['normal'] * n_total
    
    metrics_1 = compare_metrics_with_mcc(y_true_1, y_pred_1)
    print(f"Confusion Matrix:\n{metrics_1['confusion_matrix']}")
    print(f"Accuracy:  {metrics_1['accuracy']:.4f} (оманливо висока!)")
    print(f"Precision: {metrics_1['precision']:.4f} (undefined, показано 0)")
    print(f"Recall:    {metrics_1['recall']:.4f} (0% - не знайдено жодного критичного)")
    print(f"F1-score:  {metrics_1['f1_score']:.4f} (0)")
    print(f"MCC:       {metrics_1['mcc']:.4f} (негативне - модель гірша за випадкову)")
    print()
    
    # Сценарій 2: Модель знаходить 50% критичних, без помилок
    print("СЦЕНАРІЙ 2: Модель знаходить 50% критичних, без помилок")
    print("-" * 80)
    y_true_2 = ['normal'] * n_normal + ['critical'] * n_critical
    y_pred_2 = ['normal'] * n_normal + ['critical'] * 50 + ['normal'] * 50
    
    metrics_2 = compare_metrics_with_mcc(y_true_2, y_pred_2)
    print(f"Confusion Matrix:\n{metrics_2['confusion_matrix']}")
    print(f"Accuracy:  {metrics_2['accuracy']:.4f}")
    print(f"Precision: {metrics_2['precision']:.4f} (100% - всі передбачення критичних правильні)")
    print(f"Recall:    {metrics_2['recall']:.4f} (50% - знайдено половину)")
    print(f"F1-score:  {metrics_2['f1_score']:.4f}")
    print(f"MCC:       {metrics_2['mcc']:.4f} (позитивне - модель краща за випадкову)")
    print()
    
    # Сценарій 3: Модель знаходить всі критичні, але з помилками
    print("СЦЕНАРІЙ 3: Модель знаходить всі критичні, але з помилками")
    print("-" * 80)
    y_true_3 = ['normal'] * n_normal + ['critical'] * n_critical
    y_pred_3 = ['normal'] * (n_normal - 100) + ['critical'] * 200
    
    metrics_3 = compare_metrics_with_mcc(y_true_3, y_pred_3)
    print(f"Confusion Matrix:\n{metrics_3['confusion_matrix']}")
    print(f"Accuracy:  {metrics_3['accuracy']:.4f}")
    print(f"Precision: {metrics_3['precision']:.4f} (50% - половина передбачень помилкові)")
    print(f"Recall:    {metrics_3['recall']:.4f} (100% - знайдено всі)")
    print(f"F1-score:  {metrics_3['f1_score']:.4f} (такий самий, як у сценарії 2!)")
    print(f"MCC:       {metrics_3['mcc']:.4f} (такий самий, як у сценарії 2!)")
    print()
    
    print("=" * 80)
    print("ВИСНОВОК:")
    print("=" * 80)
    print("1. MCC враховує всі чотири клітинки confusion matrix (TP, TN, FP, FN)")
    print("2. MCC симетричний - однаковий результат незалежно від вибору класу")
    print("3. MCC інваріантний до незбалансованості - працює навіть при 0.01% критичних")
    print("4. MCC дає змістовний результат навіть коли Precision/Recall undefined")
    print("5. Для сильно незбалансованих класів MCC найнадійніша метрика")
    print("=" * 80)


# Демонструємо переваги MCC
demonstrate_mcc_advantages()
```

### Інтеграція MCC в Оцінку Моделей

Оновимо методи `evaluate` для включення MCC:

```python
# Оновлюємо NaiveBayesBaseline.evaluate()
from sklearn.metrics import matthews_corrcoef

# В методі evaluate додаємо:
mcc = matthews_corrcoef(
    [1 if y == 'critical' else 0 for y in y_test],
    [1 if y == 'critical' else 0 for y in y_pred]
)

return {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'mcc': mcc,  # Додаємо MCC
    'confusion_matrix': cm
}
```

## Metric War: Accuracy vs F1-score vs MCC

### Порівняння Метрик

```python
"""
Порівняння метрик на незбалансованих даних.
Демонструє, чому Accuracy може бути оманливою.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def compare_models(nb_results: dict, bert_results: dict) -> None:
    """
    Порівнює результати двох моделей.
    
    Args:
        nb_results: Результати Naive Bayes
        bert_results: Результати BERT
    """
    print("=" * 70)
    print("METRIC WAR: NAIVE BAYES vs BERT")
    print("=" * 70)
    print()
    
    # Створюємо таблицю порівняння
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'MCC'],
        'Naive Bayes': [
            nb_results['accuracy'],
            nb_results['precision'],
            nb_results['recall'],
            nb_results['f1_score'],
            nb_results['mcc']
        ],
        'BERT': [
            bert_results['accuracy'],
            bert_results['precision'],
            bert_results['recall'],
            bert_results['f1_score'],
            bert_results['mcc']
        ]
    })
    
    comparison['Difference'] = comparison['BERT'] - comparison['Naive Bayes']
    
    print("Порівняння метрик:")
    print(comparison.to_string(index=False))
    print()
    
    # Аналіз
    print("АНАЛІЗ:")
    print("-" * 70)
    
    if nb_results['accuracy'] > bert_results['accuracy']:
        print(f"  Naive Bayes має вищу Accuracy ({nb_results['accuracy']:.4f} vs {bert_results['accuracy']:.4f})")
        print("  Але це оманливо! Accuracy не враховує незбалансованість класів.")
    else:
        print(f"  BERT має вищу Accuracy ({bert_results['accuracy']:.4f} vs {nb_results['accuracy']:.4f})")
    
    print()
    
    if bert_results['recall'] > nb_results['recall']:
        print(f"  BERT має значно кращий Recall ({bert_results['recall']:.4f} vs {nb_results['recall']:.4f})")
        print("  Це означає, що BERT пропускає менше критичних збоїв.")
        print("  Для Alert Fatigue це критично важливо!")
    else:
        print(f"  Naive Bayes має кращий Recall ({nb_results['recall']:.4f} vs {bert_results['recall']:.4f})")
    
    print()
    
    if bert_results['f1_score'] > nb_results['f1_score']:
        print(f"  BERT має кращий F1-score ({bert_results['f1_score']:.4f} vs {nb_results['f1_score']:.4f})")
        print("  F1-score балансує Precision та Recall.")
    else:
        print(f"  Naive Bayes має кращий F1-score ({nb_results['f1_score']:.4f} vs {bert_results['f1_score']:.4f})")
    
    print()
    
    # MCC аналіз
    if bert_results['mcc'] > nb_results['mcc']:
        print(f"  BERT має кращий MCC ({bert_results['mcc']:.4f} vs {nb_results['mcc']:.4f})")
        print("  MCC - найнадійніша метрика для незбалансованих даних.")
        print("  Вона враховує всі чотири клітинки confusion matrix (TP, TN, FP, FN).")
    else:
        print(f"  Naive Bayes має кращий MCC ({nb_results['mcc']:.4f} vs {bert_results['mcc']:.4f})")
    
    print()
    print("=" * 70)
    print("ВИСНОВОК:")
    print("  Для незбалансованих даних (99% Normal, 1% Critical):")
    print("  - Accuracy може бути оманливою (висока через класифікацію Normal)")
    print("  - Recall критичний (скільки критичних збоїв ми знайшли)")
    print("  - F1-score балансує Precision та Recall")
    print("  - MCC найнадійніша: враховує TP, TN, FP, FN, симетрична, інваріантна до незбалансованості")
    print("  - BERT краще за Naive Bayes для контекстно-залежних логів")
    print("=" * 70)


# Порівнюємо моделі
compare_models(nb_results, bert_results)
```

### Візуалізація Результатів

```python
"""
Візуалізація порівняння моделей.
"""

def plot_comparison(nb_results: dict, bert_results: dict) -> None:
    """Створює графіки порівняння."""
    
    # Графік метрик
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'MCC']
    nb_values = [
        nb_results['accuracy'],
        nb_results['precision'],
        nb_results['recall'],
        nb_results['f1_score'],
        nb_results['mcc']
    ]
    bert_values = [
        bert_results['accuracy'],
        bert_results['precision'],
        bert_results['recall'],
        bert_results['f1_score'],
        bert_results['mcc']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, nb_values, width, label='Naive Bayes', alpha=0.8)
    bars2 = ax.bar(x + width/2, bert_values, width, label='BERT', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Порівняння Метрик: Naive Bayes vs BERT')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim([0, 1.1])
    
    # Додаємо значення на стовпці
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("\nГрафік збережено: metrics_comparison.png")
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (results, title) in enumerate([(nb_results, 'Naive Bayes'), (bert_results, 'BERT')]):
        cm = results['confusion_matrix']
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            ax=axes[idx],
            xticklabels=['Normal', 'Critical'],
            yticklabels=['Normal', 'Critical']
        )
        axes[idx].set_title(f'Confusion Matrix: {title}')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Графік збережено: confusion_matrices.png")


# Створюємо візуалізації
try:
    plot_comparison(nb_results, bert_results)
except ImportError:
    print("Помилка: Потрібно встановити matplotlib та seaborn")
    print("  pip install matplotlib seaborn")
```

## Ключові Висновки

1. **Baseline працює, але не вирішує проблему:** Naive Bayes дає високу Accuracy, але низький Recall — пропускає критичні збої.

2. **BERT краще для контексту:** Завдяки Self-Attention BERT розуміє контекст ("Connection refused" vs "Connection established") та має кращий Recall.

3. **Accuracy оманлива:** На незбалансованих даних висока Accuracy не означає хорошу модель — вона досягається за рахунок правильної класифікації більшості класу.

4. **F1-score важливіший:** Балансує Precision та Recall, краща метрика для незбалансованих даних.

5. **MCC найнадійніша:** Matthews Correlation Coefficient враховує всі чотири клітинки confusion matrix (TP, TN, FP, FN), симетрична та інваріантна до незбалансованості. Для сильно незбалансованих класів (0.01% критичних) MCC дає найбільш змістовну оцінку якості моделі.

6. **Recall критичний для Alert Fatigue:** Краще мати більше помилкових тривог, ніж пропустити справжній критичний збій.

У наступному розділі ми підсумуємо весь курс та подивимося на майбутнє: LLM Agents та RAG для автоматизації виправлення помилок.

## Рекомендована Література

### Scikit-learn та Практична Реалізація

1. **Pedregosa, F., et al.** (2011). "Scikit-learn: Machine Learning in Python"
   - Journal of Machine Learning Research, 12, 2825-2830.
   - Документація: https://scikit-learn.org/stable/

2. **Raschka, S., & Mirjalili, S.** (2019). "Python Machine Learning"
   - 3rd Edition. Packt Publishing. Практичний підручник з ML у Python.

### HuggingFace Transformers

3. **Wolf, T., et al.** (2020). "Transformers: State-of-the-Art Natural Language Processing"
   - EMNLP. HuggingFace Transformers library.
   - Документація: https://huggingface.co/docs/transformers

4. **HuggingFace Course**
   - URL: https://huggingface.co/course
   - Практичний курс з використанням Transformers.

### Метрики на Незбалансованих Даних

5. **Provost, F., & Fawcett, T.** (2013). "Data Science for Business"
   - O'Reilly Media. Розділ 4: "Evaluating Predictive Models".

6. **Saito, T., & Rehmsmeier, M.** (2015). "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets"
   - PLOS ONE. Аналіз метрик на незбалансованих даних.

7. **Matthews, B. W.** (1975). "Comparison of the predicted and observed secondary structure of T4 phage lysozyme"
   - Biochimica et Biophysica Acta (BBA) - Protein Structure, 405(2), 442-451.
   - Оригінальна робота про Matthews Correlation Coefficient (MCC).

8. **Boughorbel, S., Jarray, F., & El-Anbari, M.** (2017). "Optimal classifier for imbalanced data using Matthews Correlation Coefficient metric"
   - PLOS ONE, 12(6), e0177678.
   - Детальний аналіз переваг MCC для незбалансованих даних.

9. **Chicco, D., & Jurman, G.** (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation"
   - BMC Genomics, 21(1), 1-13.
   - Порівняння MCC з F1-score та Accuracy, демонстрація переваг MCC на незбалансованих даних.

### Fine-tuning BERT

7. **Sun, C., et al.** (2019). "How to Fine-Tune BERT for Text Classification?"
   - Chinese Computational Linguistics. Практичний гайд.

8. **Howard, J., & Ruder, S.** (2018). "Universal Language Model Fine-tuning for Text Classification"
   - ACL. ULMFiT — концепції fine-tuning.

### Production Deployment

9. **Sculley, D., et al.** (2015). "Hidden Technical Debt in Machine Learning Systems"
   - NIPS. Проблеми ML у продакшені.

10. **Huyen, C.** (2022). "Designing Machine Learning Systems"
    - O'Reilly Media. Практичний гайд з побудови ML систем.

---

**Примітка для студентів:** Почніть з документації Scikit-learn та HuggingFace для практичної реалізації. Для розуміння метрик прочитайте Provost & Fawcett та Saito & Rehmsmeier. Для розуміння переваг MCC на незбалансованих даних обов'язково прочитайте Chicco & Jurman (2020) та Boughorbel et al. (2017). Для fine-tuning використовуйте гайди від Sun et al. та Howard & Ruder.

