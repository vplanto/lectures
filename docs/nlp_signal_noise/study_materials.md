# Матеріали для опрацювання та план самостійного вивчення

## 1. Матеріали для опрацювання (Теоретичний блок)

Наступні модулі представлені у спрощеному форматі для академічного вивчення та дослідження:

### • Case Study: Alert Fatigue - Чому `grep "Error"` створює 82 години марної роботи на тиждень

**Проблема:** Система моніторингу генерує 10,000 логів на годину. Простий фільтр `grep "Error"` виявляє 1,000 рядків, але лише 10 з них (1%) є дійсно критичними. Інженер витрачає **82+ години на тиждень** на перевірку некритичних помилок.

**Математична формалізація:** Base Rate Fallacy показує, що навіть якщо лог містить "Error", ймовірність того, що він критичний, становить лише **0.95%** через високий base rate нормальних логів.

**Рішення:** NLP-фільтр (BERT-based) зменшує час перевірки з 82+ годин до **2.1 години на тиждень** (96% економія) завдяки високій Precision (85%).

**Детальніше:** 
- [01_noise_in_production.md](./01_noise_in_production.md) — Base Rate Fallacy в DevOps та Alert Fatigue
- [00_the_bayesian_trap.md](./00_the_bayesian_trap.md) — Пастка Байєса та формула повної ймовірності
- Практична симуляція: див. розділ "Практичне Завдання" нижче

### • Генератор синтетичного хаосу: Створення незбалансованого датасету для тестування

**Проблема:** Доступ до реальних логів обмежений через конфіденційність (IP-адреси, паролі, токени).

**Рішення:** Генератор синтетичних логів дозволяє створити незбалансований датасет (99% нормальних подій, 1% помилок) для тестування алгоритмів без ризику витоку даних.

**Детальніше:** 
- [06_synthetic_chaos_generator.md](./06_synthetic_chaos_generator.md) — детальний опис генерації синтетичних даних
- Практичне завдання: див. розділ "План самостійного вивчення" нижче

### • MCC як "Золотий стандарт": Чому Matthews Correlation Coefficient краще за F1-score

**Проблема:** F1-score не враховує True Negatives (TN) та може бути оманливим на незбалансованих даних.

**Рішення:** MCC враховує всі чотири зони матриці помилок (TP, TN, FP, FN) та є стійким до незбалансованості класів. Діапазон: $[-1, +1]$, де $MCC = 0$ означає випадкову класифікацію.

**Математична формалізація:**

$$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

**Детальніше:** 
- [10_seminar_metrics_extreme_imbalance.md](./10_seminar_metrics_extreme_imbalance.md) — метрики для незбалансованих даних
- Практичне порівняння: див. розділ "Практичне Застосування MCC" нижче

---

## 2. План самостійного вивчення

> **Аналогія для студента:** Самостійне вивчення цих матеріалів — це як політ за приладами в тумані. Ваші прилади — це метрики (MCC) та візуалізація (Attention/UMAP). Якщо ви навчитеся довіряти математичним показникам більше, ніж інтуїції, ви зможете знайти сигнал у будь-якому шумі.

### Модуль №1: "Alert Fatigue та Base Rate Fallacy"

**Навігація: Від візуального до аналітичного**

1. **Почніть з візуального розуміння:**
   - **Veritasium. "The Bayesian Trap"** (2021) — https://www.youtube.com/watch?v=R13BD8qKeTg
     - Класичне пояснення парадоксу медичного тесту з візуалізацією
   - **3Blue1Brown. "Bayes theorem, and making probability intuitive"** (2020) — https://www.youtube.com/watch?v=HZGCoVF3YvM
     - Геометрична інтерпретація теореми Байєса через діаграми Венна

2. **Після формування візуальної інтуїції — переходимо до математики:**
   - Розбір проблеми Alert Fatigue: чому простий `grep "Error"` створює 82+ години марної роботи

**Матеріали:** 
- [01_noise_in_production.md](./01_noise_in_production.md) — Alert Fatigue в DevOps
- [00_the_bayesian_trap.md](./00_the_bayesian_trap.md) — математична основа Base Rate Fallacy

• Математична формалізація через теорему Байєса: обчислення $P(\text{Critical} | \text{"Error"})$

**Матеріали:** 
- [02_math_setup_classification.md](./02_math_setup_classification.md) — формалізація задачі класифікації
- Практична симуляція Alert Fatigue (код нижче)

**🔍 Точка самоконтролю (Checkpoint):**

Перш ніж рухатися далі, відповідь сам собі на запитання:

> **"Якщо базова частота (base rate) події зменшується в 10 разів, як це змінить вашу Precision при незмінній точності тесту?"**

*Підказка: Використайте формулу Байєса та симулюйте зміну base rate у коді нижче.*

**Зв'язок з курсовою роботою:**
- Цей модуль є базою для теми курсової **"Адаптація BERT для детекції Concept Drift у технічних логах"** (див. розділ 5)

• Практичне завдання: Симуляція Alert Fatigue та порівняння grep vs NLP фільтра

**Код для виконання:**

```python
# Детальний код дивіться в розділі "Практичне Завдання: Симуляція Alert Fatigue"
# файлу 13_workshop_signal_in_noise.md (якщо він зберігається для довідки)
# Або використайте клас AlertFatigueSimulator з практичним завданням нижче
```

### Модуль №2: "Генератор синтетичного хаосу" (Sandbox-середовище)

**🎯 Мета модуля:** Створити контрольоване середовище для експериментів, де ви можете "погратися" з параметрами та побачити емпіричне розуміння стійкості метрик без пояснень викладача.

• Створення незбалансованого датасету (99% Normal, 1% Critical) для тестування алгоритмів

**Матеріали:** 
- [06_synthetic_chaos_generator.md](./06_synthetic_chaos_generator.md) — детальний опис генерації
- Базовий генератор (код нижче)

**🧪 Експериментальне завдання (Sandbox):**

> **"Змініть частку критичних логів з 1% до 0.01% у вашому коді і подивіться, як 'розвалиться' F1-score порівняно з MCC. Це дасть вам емпіричне розуміння стійкості метрик."**

**Інструкція:**
1. Запустіть генератор з `normal_rate=0.99` (1% критичних)
2. Навчіть модель та обчисліть F1-score та MCC
3. Змініть `normal_rate=0.9999` (0.01% критичних)
4. Порівняйте зміни в метриках
5. Візуалізуйте результати (див. Self-Diagnostic Scripts нижче)

• Розширення з контекстною залежністю: "Connection refused" (Critical) vs "Connection established" (Normal)

**Матеріали:** 
- [05_bert_and_transformers.md](./05_bert_and_transformers.md) — чому контекст важливий для BERT
- [03_naive_bayes_deep_dive.md](./03_naive_bayes_deep_dive.md) — межі Naive Bayes для контекстно-залежних логів

**Зв'язок з курсовою роботою:**
- Цей модуль є основою для теми курсової **"Порівняння Multinomial та Bernoulli Naive Bayes для фільтрації технічних логів"** (див. розділ 5)

**Практичне завдання:**

```python
"""
Базовий генератор синтетичних логів для гуртка.
"""
import random
from typing import List, Tuple
import pandas as pd

class SyntheticLogGenerator:
    """
    Генератор синтетичних технічних логів.
    """
    
    def __init__(self, normal_rate: float = 0.99):
        self.normal_rate = normal_rate
        
        # Шаблони для нормальних логів
        self.normal_templates = [
            "Request processed successfully",
            "Connection established",
            "User login successful",
            "Data synchronized",
            "Cache updated",
            "Service started",
            "Operation completed",
            "Transaction committed",
            "File uploaded",
            "Query executed"
        ]
        
        # Шаблони для критичних логів
        self.critical_templates = [
            "Connection refused by database",
            "Authentication failed: invalid credentials",
            "Database connection timeout",
            "Memory allocation error",
            "Disk space exhausted",
            "Network interface down",
            "Service unavailable",
            "Transaction rollback due to deadlock",
            "Critical security breach detected",
            "System crash: kernel panic"
        ]
    
    def generate_log(self) -> Tuple[str, int]:
        is_critical = random.random() > self.normal_rate
        
        if is_critical:
            template = random.choice(self.critical_templates)
            label = 1
        else:
            template = random.choice(self.normal_templates)
            label = 0
        
        timestamp = f"[2024-01-{random.randint(1, 28)} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}]"
        log_text = f"{timestamp} {template}"
        
        return log_text, label
    
    def generate_dataset(self, n: int = 10000) -> pd.DataFrame:
        logs = []
        labels = []
        
        for _ in range(n):
            log, label = self.generate_log()
            logs.append(log)
            labels.append(label)
        
        return pd.DataFrame({'log': logs, 'label': labels})

# Приклад використання
if __name__ == "__main__":
    generator = SyntheticLogGenerator(normal_rate=0.99)
    df = generator.generate_dataset(n=10000)
    
    print(f"Normal: {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)")
    print(f"Critical: {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)")
```

### Модуль №3: "MCC vs F1-Score на незбалансованих даних"

**Навігація: Від візуального до аналітичного**

1. **Почніть з візуального розуміння:**
   - Рекомендується спочатку подивитися візуалізації confusion matrix та порівняння метрик у практичному коді нижче
   - Запустіть Self-Diagnostic Script для візуалізації (див. розділ "Автоматизація перевірки")

2. **Після формування візуальної інтуїції — переходимо до математики:**
   - Порівняння MCC та F1-score: чому MCC краще для технічних доменів

**Матеріали:** 
- [10_seminar_metrics_extreme_imbalance.md](./10_seminar_metrics_extreme_imbalance.md) — метрики для незбалансованих даних
- Практичне порівняння (код нижче)

• Інтерпретація MCC: $MCC > 0.5$ (хороша модель), $MCC > 0.7$ (відмінна модель)

**🔍 Точка самоконтролю (Checkpoint):**

Перш ніж рухатися далі, відповідь сам собі на запитання:

> **"Чому при Accuracy 99.9% ваш MCC може дорівнювати 0? Що це каже про вашу модель?"**

*Підказка: Подумайте про модель, яка завжди передбачає більшість клас. Запустіть код нижче з `y_pred_always_normal` та подивіться на результати.*

**Зв'язок з курсовою роботою:**
- Цей модуль є основою для теми курсової **"Оптимізація BERT для real-time моніторингу через квантизацію та дистиляцію"** (див. розділ 5)

**Практичне завдання:**

```python
"""
Порівняння MCC та F1-score на незбалансованих даних.
"""
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix

# Симулюємо незбалансовані дані (99% Normal, 1% Critical)
n_samples = 10000
n_critical = 100  # 1%
n_normal = 9900   # 99%

y_true = np.concatenate([
    np.ones(n_critical),  # Critical = 1
    np.zeros(n_normal)   # Normal = 0
])

# Сценарій: Модель, яка завжди передбачає Normal
y_pred_always_normal = np.zeros(n_samples)

# Обчислюємо метрики
mcc = matthews_corrcoef(y_true, y_pred_always_normal)
f1 = f1_score(y_true, y_pred_always_normal)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_always_normal).ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"Accuracy: {accuracy:.2%}")  # 99% - оманлива!
print(f"F1-Score: {f1:.4f}")        # 0.00 - правильно показує проблему
print(f"MCC: {mcc:.4f}")             # 0.00 - правильно показує випадкову класифікацію
```

---

## 3. Автоматизація перевірки (Self-Diagnostic Scripts)

Для самостійного навчання критично мати скрипти, які візуалізують помилки та допомагають зрозуміти принципи роботи моделей.

### 3.1. Візуалізація Attention Weights для BERT

**Мета:** Коли студент сам побачить на графіку, що модель "дивиться" на слово `refused` як на головний сигнал, він зрозуміє принцип Self-Attention краще, ніж з формул.

**Скрипт для візуалізації:**

```python
"""
Self-Diagnostic Script: Візуалізація Attention Weights для BERT
"""
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(text, model, tokenizer, layer=0, head=0):
    """
    Візуалізує Attention weights для заданого тексту.
    
    Args:
        text: Вхідний текст (наприклад, "Connection refused by database")
        model: Fine-tuned BERT модель
        tokenizer: BERT tokenizer
        layer: Номер шару (0-11 для BERT-base)
        head: Номер голови (0-11 для BERT-base)
    """
    # Токенізація
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Отримання Attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # Tuple з 12 шарами
    
    # Витягуємо Attention для конкретного шару та голови
    attention = attentions[layer][0, head].numpy()  # [seq_len, seq_len]
    
    # Отримуємо токени для підписів
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Візуалізація
    plt.figure(figsize=(12, 8))
    sns.heatmap(attention, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Attention Weight'})
    plt.title(f'Attention Weights - Layer {layer}, Head {head}\nText: "{text}"')
    plt.xlabel('Key (What to attend to)')
    plt.ylabel('Query (What is attending)')
    plt.tight_layout()
    plt.show()
    
    # Аналіз: Знаходимо слово з найвищою увагою
    max_attention_idx = np.argmax(attention.sum(axis=0))
    print(f"\n🔍 Найбільшу увагу привертає слово: '{tokens[max_attention_idx]}'")
    print(f"   Це підтверджує, що BERT використовує контекст для класифікації.")

# Приклад використання
if __name__ == "__main__":
    # Завантажуємо модель (fine-tuned на логах)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Тестуємо на критичному лозі
    critical_log = "Connection refused by database server"
    visualize_attention(critical_log, model, tokenizer, layer=0, head=0)
    
    # Порівнюємо з нормальним логом
    normal_log = "Connection established successfully"
    visualize_attention(normal_log, model, tokenizer, layer=0, head=0)
```

**Що показує цей скрипт:**
- Які слова модель вважає найважливішими для класифікації
- Як Self-Attention зважує різні частини речення
- Чому BERT розрізняє "refused" та "established" навіть з однаковим словом "Connection"

### 3.2. Візуалізація квантизації ваг (INT8)

**Мета:** Дати скрипт, що малює розподіл ваг до і після переходу в INT8. Це візуальне підтвердження "дискретизації" знань.

**Скрипт для візуалізації:**

```python
"""
Self-Diagnostic Script: Візуалізація квантизації ваг BERT
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertModel

def visualize_quantization(model_fp32, model_int8):
    """
    Візуалізує розподіл ваг до та після квантизації.
    
    Args:
        model_fp32: Модель у форматі FP32
        model_int8: Квантизована модель у форматі INT8
    """
    # Збираємо ваги з усіх шарів
    weights_fp32 = []
    weights_int8 = []
    
    for name, param in model_fp32.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            weights_fp32.append(param.data.flatten().cpu().numpy())
    
    for name, param in model_int8.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            # Конвертуємо INT8 назад у FP32 для візуалізації
            weights_int8.append(param.data.float().flatten().cpu().numpy())
    
    weights_fp32 = np.concatenate(weights_fp32)
    weights_int8 = np.concatenate(weights_int8)
    
    # Візуалізація
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # До квантизації (FP32)
    axes[0].hist(weights_fp32, bins=100, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Розподіл ваг ДО квантизації (FP32)')
    axes[0].set_xlabel('Значення ваги')
    axes[0].set_ylabel('Частота')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(weights_fp32.mean(), color='red', linestyle='--', label=f'Середнє: {weights_fp32.mean():.4f}')
    axes[0].legend()
    
    # Після квантизації (INT8)
    axes[1].hist(weights_int8, bins=100, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Розподіл ваг ПІСЛЯ квантизації (INT8)')
    axes[1].set_xlabel('Значення ваги')
    axes[1].set_ylabel('Частота')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(weights_int8.mean(), color='red', linestyle='--', label=f'Середнє: {weights_int8.mean():.4f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Статистика
    print(f"\n📊 Статистика квантизації:")
    print(f"   FP32: Мін={weights_fp32.min():.4f}, Макс={weights_fp32.max():.4f}, Std={weights_fp32.std():.4f}")
    print(f"   INT8: Мін={weights_int8.min():.4f}, Макс={weights_int8.max():.4f}, Std={weights_int8.std():.4f}")
    print(f"   Втрата інформації: {((weights_fp32.std() - weights_int8.std()) / weights_fp32.std() * 100):.2f}%")
    
    # Візуалізація помилки квантизації
    if len(weights_fp32) == len(weights_int8):
        quantization_error = np.abs(weights_fp32 - weights_int8)
        plt.figure(figsize=(10, 6))
        plt.hist(quantization_error, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Розподіл помилки квантизації (|FP32 - INT8|)')
        plt.xlabel('Абсолютна помилка')
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        plt.show()

# Приклад використання
if __name__ == "__main__":
    # Завантажуємо модель FP32
    model_fp32 = BertModel.from_pretrained('bert-base-uncased')
    
    # Квантизуємо (приклад - реальна квантизація потребує спеціальних бібліотек)
    # model_int8 = quantize_model(model_fp32)  # Використайте torch.quantization або інші інструменти
    
    # Візуалізуємо
    # visualize_quantization(model_fp32, model_int8)
```

**Що показує цей скрипт:**
- Як квантизація "дискретизує" неперервні значення ваг
- Втрату інформації через обмеження точності
- Розподіл помилки квантизації

### 3.3. Візуалізація метрик на незбалансованих даних

**Скрипт для порівняння MCC vs F1-Score:**

```python
"""
Self-Diagnostic Script: Візуалізація метрик на незбалансованих даних
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix
import seaborn as sns

def visualize_metrics_comparison(y_true, y_pred, model_name="Model"):
    """
    Візуалізує порівняння різних метрик для незбалансованих даних.
    """
    # Обчислюємо метрики
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Візуалізація
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Normal', 'Critical'],
                yticklabels=['Normal', 'Critical'])
    axes[0].set_title(f'Confusion Matrix - {model_name}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Метрики
    metrics = ['Accuracy', 'F1-Score', 'MCC']
    values = [accuracy, f1, mcc]
    colors = ['green' if v > 0.5 else 'orange' if v > 0.3 else 'red' for v in values]
    
    bars = axes[1].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Значення метрики')
    axes[1].set_title(f'Порівняння метрик - {model_name}')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Додаємо значення на стовпці
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Виводимо інтерпретацію
    print(f"\n📊 Інтерпретація метрик для {model_name}:")
    print(f"   Accuracy: {accuracy:.2%} - {'✅ Хороша' if accuracy > 0.9 else '⚠️ Оманлива на незбалансованих даних'}")
    print(f"   F1-Score: {f1:.3f} - {'✅ Хороша' if f1 > 0.5 else '⚠️ Низька'}")
    print(f"   MCC: {mcc:.3f} - ", end='')
    if mcc > 0.7:
        print("✅ Відмінна модель!")
    elif mcc > 0.5:
        print("✅ Хороша модель")
    elif mcc > 0.3:
        print("⚠️ Модель потребує покращення")
    else:
        print("❌ Модель не працює добре (близька до випадкової)")

# Приклад використання
if __name__ == "__main__":
    # Симулюємо незбалансовані дані
    n_samples = 10000
    n_critical = 100  # 1%
    n_normal = 9900   # 99%
    
    y_true = np.concatenate([
        np.ones(n_critical),  # Critical = 1
        np.zeros(n_normal)   # Normal = 0
    ])
    
    # Сценарій 1: Модель завжди передбачає Normal
    y_pred_always_normal = np.zeros(n_samples)
    visualize_metrics_comparison(y_true, y_pred_always_normal, "Always Normal")
    
    # Сценарій 2: Модель з помилками
    y_pred_with_errors = y_true.copy()
    # Додаємо 5% помилок
    error_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    y_pred_with_errors[error_indices] = 1 - y_pred_with_errors[error_indices]
    visualize_metrics_comparison(y_true, y_pred_with_errors, "Model with Errors")
```

---

## 4. Кейси для фінальної презентації: Три траєкторії

### Варіант 1: Класичний (Naive Bayes) - Фільтр Спаму в Логах

**Мета:** Побудувати фільтр спаму (неважливих логів) за допомогою Naive Bayes.

**Завдання:**
- Порівняння Multinomial та Bernoulli моделей Naive Bayes
- Особливості технічних логів (IP-адреси, хеші, UUID, структуровані формати)
- Використання MCC як основної метрики

**Базові матеріали:** 
- [03_naive_bayes_deep_dive.md](./03_naive_bayes_deep_dive.md) — Naive Bayes та його межі
- [02_math_setup_classification.md](./02_math_setup_classification.md) — математична формалізація
- [07_implementation_workshop.md](./07_implementation_workshop.md) — практична реалізація

**Структура презентації (Шаблон дослідження):**

1. **Проблема:** Чому `grep` або базовий метод тут провалився?
   - Опис Alert Fatigue та втрати часу
   - Чому простий фільтр не працює

2. **Математична гіпотеза:** Яку властивість ми використовуємо?
   - Незалежність ознак у Naive Bayes
   - Припущення про розподіл слів у логах

3. **Експеримент:** Порівняння моделей на синтетичних даних
   - Multinomial vs Bernoulli Naive Bayes
   - Метрики: MCC, F1-Score, Accuracy
   - Візуалізація результатів (див. Self-Diagnostic Scripts)

4. **Аналіз помилок:** Візуалізація через confusion matrix
   - Які логи класифікуються неправильно?
   - Чому одна модель краща за іншу?

5. **Висновки:** Яка модель краща для технічних логів?

### Варіант 2: Семантичний (BERT) - Детекція Аномалій у Логах Kubernetes

**Мета:** Використати BERT для детекції аномалій у логах Kubernetes, де важливий контекст.

**Завдання:**
- Контекстна залежність: "Connection established" (Normal) vs "Connection refused" (Critical)
- Fine-tuning BERT на синтетичних логах Kubernetes
- Візуалізація Attention weights для пояснення рішень

**Базові матеріали:** 
- [05_bert_and_transformers.md](./05_bert_and_transformers.md) — BERT та Self-Attention, візуалізація Attention
- [04_geometry_of_meaning.md](./04_geometry_of_meaning.md) — від слів до векторів
- [07_implementation_workshop.md](./07_implementation_workshop.md) — практична реалізація BERT

**Структура презентації (Шаблон дослідження):**

1. **Проблема:** Чому `grep` або базовий метод тут провалився?
   - Контекстна залежність: "Connection established" vs "Connection refused"
   - Чому Naive Bayes не розрізняє ці логи?

2. **Математична гіпотеза:** Яку властивість ми використовуємо?
   - Контекстну залежність у BERT через Self-Attention
   - Як BERT кодує контекст у embeddings

3. **Експеримент:** Порівняння моделей на синтетичних даних
   - BERT vs Naive Bayes на логах з контекстною залежністю
   - Метрики: MCC, F1-Score, Precision, Recall

4. **Аналіз помилок:** Візуалізація через Attention maps та t-SNE/UMAP
   - **Обов'язково:** Візуалізація Attention weights (див. Self-Diagnostic Script 3.1)
   - Які слова модель вважає найважливішими?
   - Чому BERT розрізняє "refused" та "established"?

5. **Висновки:** Переваги BERT для контекстно-залежних логів

### Варіант 3: Оптимізаційний (DistilBERT) - Швидкий AI-Моніторинг

**Мета:** Зробити AI-моніторинг швидким для високонавантажених систем через квантизацію ваг.

**Завдання:**
- Квантизація моделі: FP32 → INT8 з візуалізацією помилки квантизації
- Бенчмарк швидкості: BERT vs DistilBERT vs квантизовані моделі
- Баланс якості та швидкості для production

**Базові матеріали:** 
- [11_seminar_efficient_nlp_quantization_distillation.md](./11_seminar_efficient_nlp_quantization_distillation.md) — квантизація та дистиляція, візуалізація помилки квантизації
- [05_bert_and_transformers.md](./05_bert_and_transformers.md) — обчислювальна складність BERT

**Структура презентації (Шаблон дослідження):**

1. **Проблема:** Чому `grep` або базовий метод тут провалився?
   - Проблема швидкості BERT для production
   - Чому повна BERT модель занадто повільна для real-time моніторингу?

2. **Математична гіпотеза:** Яку властивість ми використовуємо?
   - Стійкість MCC до незбалансованості
   - Як квантизація зберігає важливі патерни в вагах

3. **Експеримент:** Порівняння моделей на синтетичних даних
   - Бенчмарк швидкості: BERT vs DistilBERT vs квантизовані моделі
   - Бенчмарк якості: MCC, F1-Score на незбалансованих даних

4. **Аналіз помилок:** Візуалізація через розподіл ваг та метрики
   - **Обов'язково:** Візуалізація квантизації ваг (див. Self-Diagnostic Script 3.2)
   - Розподіл ваг до та після квантизації
   - Втрата інформації через дискретизацію

5. **Висновки:** Оптимальна модель для production моніторингу
   - Баланс якості (MCC) та швидкості

---

## 5. Практичне застосування MCC

### Порівняння MCC vs F1-Score на незбалансованих даних

**Ключові висновки:**

1. **MCC враховує всі чотири зони матриці помилок** (TP, TN, FP, FN)
2. **F1-score не враховує True Negatives**, що критично на незбалансованих даних
3. **"Завжди Normal" має Accuracy=99%**, але MCC=0.00 (випадкова класифікація)
4. **MCC дає повну картину якості класифікації**, незалежно від балансу класів
5. **Для технічних доменів MCC є "золотим стандартом" метрики**

**Практичний код:**

```python
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix

# Після навчання моделі
y_pred = model.predict(X_test)

# Обчислюємо MCC
mcc = matthews_corrcoef(y_test, y_pred)
print(f"MCC: {mcc:.4f}")

# Інтерпретація
if mcc > 0.7:
    print("Відмінна модель!")
elif mcc > 0.5:
    print("Хороша модель")
elif mcc > 0.3:
    print("Модель потребує покращення")
else:
    print("Модель не працює добре")
```

**Детальніше:** 
- [10_seminar_metrics_extreme_imbalance.md](./10_seminar_metrics_extreme_imbalance.md) — метрики для незбалансованих даних
- Практичне порівняння з візуалізацією (див. код вище)

---

## 6. Теми для курсових/дипломних робіт (на перспективу)

На основі цих матеріалів можна обрати напрямки для академічних досліджень:

1. **"Адаптація BERT для детекції Concept Drift у технічних логах"**

   **Зв'язок з модулями:**
   - **Модуль №1** (Base Rate Fallacy) — базове розуміння проблеми незбалансованості
   - **Модуль №2** (Генератор синтетичного хаосу) — створення даних для тестування
   - **Модуль №3** (MCC) — оцінка якості моделі на незбалансованих даних
   
   **Базові матеріали:** 
   - [08_course_summary.md](./08_course_summary.md) — Concept Drift та методи виявлення
   - [05_bert_and_transformers.md](./05_bert_and_transformers.md) — BERT архітектура
   - [06_synthetic_chaos_generator.md](./06_synthetic_chaos_generator.md) — генерація даних для тестування
   
   **Рекомендовані інструменти:**
   - Self-Diagnostic Script 3.1 (візуалізація Attention weights)
   - Self-Diagnostic Script 3.3 (візуалізація метрик)

2. **"Порівняння Multinomial та Bernoulli Naive Bayes для фільтрації технічних логів"**

   **Зв'язок з модулями:**
   - **Модуль №2** (Генератор синтетичного хаосу) — створення незбалансованого датасету для тестування
   - **Модуль №3** (MCC) — оцінка якості моделей на незбалансованих даних
   
   **Базові матеріали:**
   - [03_naive_bayes_deep_dive.md](./03_naive_bayes_deep_dive.md) — Naive Bayes та його варіанти
   - [02_math_setup_classification.md](./02_math_setup_classification.md) — математична формалізація
   - [10_seminar_metrics_extreme_imbalance.md](./10_seminar_metrics_extreme_imbalance.md) — оцінка на незбалансованих даних
   
   **Рекомендовані інструменти:**
   - Sandbox-середовище (зміна частки критичних логів)
   - Self-Diagnostic Script 3.3 (візуалізація метрик)

3. **"Оптимізація BERT для real-time моніторингу через квантизацію та дистиляцію"**

   **Зв'язок з модулями:**
   - **Модуль №3** (MCC) — оцінка якості після оптимізації (MCC має залишатися високим)
   - **Модуль №2** (Генератор синтетичного хаосу) — тестування на незбалансованих даних
   
   **Базові матеріали:**
   - [11_seminar_efficient_nlp_quantization_distillation.md](./11_seminar_efficient_nlp_quantization_distillation.md) — квантизація та дистиляція
   - [05_bert_and_transformers.md](./05_bert_and_transformers.md) — обчислювальна складність BERT
   - [07_implementation_workshop.md](./07_implementation_workshop.md) — практична реалізація
   
   **Рекомендовані інструменти:**
   - Self-Diagnostic Script 3.2 (візуалізація квантизації ваг)
   - Self-Diagnostic Script 3.3 (порівняння метрик до/після оптимізації)

4. **"Візуалізація Attention weights для explainability в AI-SRE системах"**

   **Зв'язок з модулями:**
   - **Модуль №2** (Генератор синтетичного хаосу) — тестування на різних типах логів
   - **Модуль №3** (MCC) — оцінка якості моделі
   
   **Базові матеріали:**
   - [05_bert_and_transformers.md](./05_bert_and_transformers.md) — Explainability та візуалізація Attention weights
   - [09_seminar_high_dimensional_geometry.md](./09_seminar_high_dimensional_geometry.md) — візуалізація високих розмірностей
   
   **Рекомендовані інструменти:**
   - Self-Diagnostic Script 3.1 (візуалізація Attention weights) — **обов'язково**
   - Self-Diagnostic Script 3.3 (візуалізація метрик)

---

## Резюме для самостійного навчання

**Найкраще, що можна зробити за відсутності семінарів — це перетворити навчання на серію контрольованих експериментів.**

Коли студент сам "зламає" модель, занадто сильно зменшивши base rate, а потім "полагодить" її через MCC або BERT, він отримає набагато глибші знання, ніж на лекції.

**Ключові принципи:**

1. **Точки самоконтролю** — не рухайтеся далі, поки не відповісте на запитання
2. **Sandbox-середовище** — експериментуйте з параметрами та спостерігайте результати
3. **Від візуального до аналітичного** — почніть з відео, потім математика
4. **Self-Diagnostic Scripts** — візуалізація допомагає краще зрозуміти принципи
5. **Шаблон дослідження** — структурований підхід до презентації результатів

---

## Академічна аналогія

**Простий `grep "Error"`** — це як інженер, який перевіряє кожен лог зі словом "Error" вручну, витрачаючи 82+ години на тиждень на перевірку некритичних помилок.

**NLP-фільтр (BERT)** — це як розумний асистент, який розуміє контекст та визначає, які логи дійсно критичні, зменшуючи час перевірки до 2.1 години на тиждень.

**MCC як "золотий стандарт"** — це як точний термометр, який показує реальну температуру (якість класифікації), на відміну від F1-score, який може бути оманливим на незбалансованих даних, як термометр, що показує "99%", коли насправді модель працює випадково.

---

## Рекомендована література

1. **Matthews, B. W.** (1975). "Comparison of the predicted and observed secondary structure of T4 phage lysozyme"
   - Biochimica et Biophysica Acta (BBA) - Protein Structure. Оригінальна робота про MCC.

2. **Chicco, D., & Jurman, G.** (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and Accuracy in binary classification evaluation"
   - BMC Genomics. Детальне порівняння MCC з іншими метриками.

3. **Boughorbel, S., Jarray, F., & El-Anbari, M.** (2017). "Optimal classifier for imbalanced data using Matthews Correlation Coefficient metric"
   - PLOS ONE. Застосування MCC для незбалансованих даних.

