---
title: "05 Bert And Transformers"
type: lecture
module: Семантика
prerequisites: module 4
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# BERT та Трансформери: Розуміння Контексту в Технічних Логах

"Connection refused" та "Connection established" містять одне слово, але мають протилежні значення. Word2Vec не розрізнить їх — обидва матимуть подібні вектори.

BERT розуміє контекст. Механізм Self-Attention дозволяє моделі "дивитися" на всі слова одночасно та зважувати їх важливість залежно від контексту.

## Від Word2Vec до Контекстних Embeddings

### Проблема статичних embeddings

**Word2Vec:** Кожне слово має один вектор незалежно від контексту.

$$
\mathbf{v}_{\texttt{"bank"}} = (0.2, -0.1, 0.5, \ldots)
$$

Цей вектор однаковий для:
- "river bank" (берег ріки)
- "bank account" (банківський рахунок)

**Проблема:** Полісемія (одне слово, різні значення) не враховується.

### Контекстні embeddings

**BERT:** Кожне слово має різний вектор залежно від контексту.

$$
\mathbf{v}_{\texttt{"bank"}}^{(1)} \neq \mathbf{v}_{\texttt{"bank"}}^{(2)}
$$

де верхній індекс позначає позицію в реченні.

**Приклад:**

```
Речення 1: "Connection refused by database"
Речення 2: "Connection established successfully"
```

Для BERT:

$$
\mathbf{v}_{\texttt{"Connection"}}^{(1)} \neq \mathbf{v}_{\texttt{"Connection"}}^{(2)}
$$

Перший вектор кодує негативний контекст (`"refused"`), другий — позитивний (`"established"`).

## Архітектура Трансформера

### Загальна Структура

Трансформер складається з:

1. **Embedding Layer** — перетворення токенів у вектори
2. **Positional Encoding** — додавання інформації про позицію
3. **Encoder Layers** (N шарів):
   - Multi-Head Self-Attention
   - Feed-Forward Network
   - Residual Connections
   - Layer Normalization
4. **Output Layer** — фінальне представлення

### Математична Формалізація

**Вхід:** Послідовність токенів $X = \{x_1, x_2, \ldots, x_n\}$

**Embedding:**

$$
\mathbf{E} = [\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n]
$$

де $\mathbf{e}_i \in \mathbb{R}^d$ — embedding токена $x_i$.

**Positional Encoding:**

$$
\mathbf{PE}_{(\text{pos}, 2i)} = \sin\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

$$
\mathbf{PE}_{(\text{pos}, 2i+1)} = \cos\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

де $\text{pos}$ — позиція токена в реченні, $i$ — індекс виміру (dimension index).

**Вхід до Encoder:**

$$
\mathbf{H}^{(0)} = \mathbf{E} + \mathbf{PE}
$$

## Self-Attention: Як Модель Розуміє Контекст

### Інтуїція

Self-Attention дозволяє кожному слову "дивитися" на всі інші слова в реченні та зважувати їх важливість.

**Приклад:**

```
"Connection refused by database server"
```

Для слова "refused":
- "Connection" — важливе (показує, що саме відмовлено)
- "database" — важливе (показує об'єкт)
- "server" — менш важливе (уточнення)
- "by" — неважливе (службове слово)

### Математична Формалізація

**Query, Key, Value:**

Для кожного слова $x_i$ обчислюємо три вектори:

$$
\begin{aligned}
\mathbf{q}_i &= \mathbf{W}_Q \mathbf{e}_i \quad &&\text{(Query)} \\
\mathbf{k}_i &= \mathbf{W}_K \mathbf{e}_i \quad &&\text{(Key)} \\
\mathbf{v}_i &= \mathbf{W}_V \mathbf{e}_i \quad &&\text{(Value)}
\end{aligned}
$$

де $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$ — матриці ваг, які навчаються.

**Attention Scores (Оцінка важливості):**

$$
\text{score}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}
$$

де $d_k$ — розмірність векторів Key (масштабування запобігає затуханню градієнтів).

**Attention Weights (Softmax):**

$$
\alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_{m=1}^{n} \exp(\text{score}_{im})}
$$

**Output (Contextual Vector):**

$$
\mathbf{h}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{v}_j
$$

Вектор $\mathbf{h}_i$ тепер містить інформацію про слово $x_i$ **з урахуванням** усіх інших слів речення (зважених за їх важливістю $\alpha_{ij}$).

### Матрична Форма

**Всі слова одночасно (матрична форма):**

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{H} \mathbf{W}_Q \\
\mathbf{K} &= \mathbf{H} \mathbf{W}_K \\
\mathbf{V} &= \mathbf{H} \mathbf{W}_V
\end{aligned}
$$

**Attention:**

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

### Self-Attention як тензорна операція: лінійна алгебра для студентів 2 курсу

**Важливо для студентів, які вивчають лінійну алгебру паралельно:** Self-Attention — це фактично послідовність операцій над тензорами (багатовимірними масивами), які можна розуміти через призму лінійної алгебри.

#### Тензорна Інтерпретація

**Вхідний тензор:** $\mathbf{H} \in \mathbb{R}^{n \times d}$
- $n$ — кількість токенів (слів) у послідовності
- $d$ — розмірність embedding (наприклад, 768 для BERT-base)
- Кожен рядок $\mathbf{H}[i, :]$ — це embedding $i$-го токена

**Проекційні матриці:** $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times d_k}$
- Це лінійні перетворення (лінійні відображення).
- Кожна матриця проектує embeddings у різний підпростір ознак.

#### "Фізичний" зміст Q, K, V: проекції у різні підпростори ознак

**1. Query ($\mathbf{Q}$): "Що я шукаю?"**
- $\mathbf{Q} = \mathbf{H} \mathbf{W}_Q$ — проекція embeddings у підпростір "запитів".
- Кожен рядок $\mathbf{q}_i$ кодує: *"Яку інформацію я (токен $i$) шукаю в інших токенах?"*
- **Геометрична інтерпретація:** $\mathbf{W}_Q$ повертає простір ознак так, щоб виділити аспекти, які токен "запитує" у контексті.

**2. Key ($\mathbf{K}$): "Що я можу запропонувати?"**
- $\mathbf{K} = \mathbf{H} \mathbf{W}_K$ — проекція embeddings у підпростір "ключів".
- Кожен рядок $\mathbf{k}_j$ кодує: *"Яку інформацію я (токен $j$) можу надати іншим токенам?"*
- **Геометрична інтерпретація:** $\mathbf{W}_K$ повертає простір ознак так, щоб виділити аспекти, які токен "пропонує" контексту.

**3. Value ($\mathbf{V}$): "Що я фактично передаю?"**
- $\mathbf{V} = \mathbf{H} \mathbf{W}_V$ — проекція embeddings у підпростір "значень".
- Кожен рядок $\mathbf{v}_j$ кодує: *"Яку фактичну інформацію я (токен $j$) передаю, якщо мене обирають?"*
- **Геометрична інтерпретація:** $\mathbf{W}_V$ повертає простір ознак так, щоб виділити аспекти, які токен "передає" при обранні.

#### Математична Формалізація Проекцій

**Лінійні перетворення:**

$$
\begin{aligned}
\mathbf{Q} &= \mathbf{H} \mathbf{W}_Q \quad &&\text{(проекція у підпростір запитів)} \\
\mathbf{K} &= \mathbf{H} \mathbf{W}_K \quad &&\text{(проекція у підпростір ключів)} \\
\mathbf{V} &= \mathbf{H} \mathbf{W}_V \quad &&\text{(проекція у підпростір значень)}
\end{aligned}
$$

**Геометрична інтерпретація:**
- $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ — це матриці повороту та масштабування простору ознак.
- Кожна матриця виділяє різні аспекти семантики токенів.
- Підпростори можуть бути ортогональними або перетинатися залежно від навчання.

**Скалярний добуток як міра відповідності:**

$$
\text{score}_{ij} = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}
$$

**Геометрична інтерпретація:**
- $\mathbf{q}_i \cdot \mathbf{k}_j$ — пропорційний косинусу кута між векторами $\mathbf{q}_i$ та $\mathbf{k}_j$ (за умови нормалізації).
- Високий score означає, що токен $j$ "відповідає" на запит токена $i$.
- **Аналогія:** Як пошук у базі даних: Query — запит, Key — індекс, Value — дані.

#### Приклад: "Connection refused by database"

**Токени:** `[CLS]`, `Connection`, `refused`, `by`, `database`

**Для токена "refused":**
- **Query:** "Я шукаю токени, які показують, що саме відмовлено"
- **Key (від "Connection"):** "Я пропоную інформацію про те, що саме з'єднання відмовлено"
- **Key (від "database"):** "Я пропоную інформацію про об'єкт відмови"
- **Value (від "Connection"):** Фактична семантика "Connection" в контексті відмови
- **Value (від "database"):** Фактична семантика "database" в контексті відмови

**Attention weights:**
- $\alpha_{\texttt{refused}, \texttt{Connection}} \approx 0.4$ (високий — `Connection` важливий для `refused`)
- $\alpha_{\texttt{refused}, \texttt{database}} \approx 0.3$ (високий — `database` важливий)
- $\alpha_{\texttt{refused}, \texttt{by}} \approx 0.1$ (низький — `by` неважливий)

**Результат (зважена сума):**

$$
\mathbf{h}_{\texttt{refused}} = 0.4 \cdot \mathbf{v}_{\texttt{Connection}} + 0.3 \cdot \mathbf{v}_{\texttt{database}} + 0.1 \cdot \mathbf{v}_{\texttt{by}} + \ldots
$$

#### Властивості проекцій

**1. Лінійність:**
- Кожна проекція — це лінійне перетворення: $f(\mathbf{x}) = \mathbf{W} \mathbf{x}$.
- Композиція проекцій також лінійна.

**2. Збереження структури:**
- Якщо два токени схожі в оригінальному просторі ($\mathbf{H}$), вони можуть залишитися схожими або стати різними в проекційних просторах залежно від матриць $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$.

**3. Незалежність підпросторів:**
- Підпростори Query, Key, Value навчаються незалежно
- Модель сама визначає, які аспекти виділяти в кожному підпросторі

#### Зв'язок з Лінійною Алгеброю

**Для студентів 2 курсу, які вивчають лінійну алгебру:**

1. **Лінійні відображення:** $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ — це матриці лінійних відображень (операторів) у векторному просторі.
2. **Власні значення та вектори:** Аналіз спектру цих матриць дозволяє зрозуміти, які компоненти вхідних даних посилюються, а які — пригнічуються.
3. **Ортогональність:** У деяких архітектурах підпростори $Q, K, V$ намагаються зробити ортогональними для зменшення надлишковості.
4. **Ранг матриці:** $\text{rank}(\mathbf{W})$ визначає ефективну розмірність підпростору ознак. Якщо $\text{rank} < d_k$, відбувається стиснення інформації (low-rank bottleneck).

**Практичне завдання для розуміння:**

```python
"""
Демонстрація проекцій Q, K, V як лінійних перетворень.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Симуляція embeddings (n=5 токенів, d=10 розмірність)
H = np.random.randn(5, 10)  # 5 токенів, 10-вимірні embeddings

# Проекційні матриці (d=10, d_k=8)
W_Q = np.random.randn(10, 8)
W_K = np.random.randn(10, 8)
W_V = np.random.randn(10, 8)

# Проекції
Q = H @ W_Q  # (5, 8) - проекція у підпростір запитів
K = H @ W_K  # (5, 8) - проекція у підпростір ключів
V = H @ W_V  # (5, 8) - проекція у підпростір значень

# Attention scores
scores = Q @ K.T / np.sqrt(8)  # (5, 5) - матриця відповідностей
attention_weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

# Output
H_output = attention_weights @ V  # (5, 8) - зважена сума значень

print("Вхідний тензор H:", H.shape)
print("Проекція Q:", Q.shape)
print("Проекція K:", K.shape)
print("Проекція V:", V.shape)
print("Attention weights:", attention_weights.shape)
print("Вихідний тензор:", H_output.shape)

# Візуалізація: як змінюються токени в різних підпросторах
# Використовуємо PCA для зниження розмірності до 2D для візуалізації
pca = PCA(n_components=2)

H_2d = pca.fit_transform(H)
Q_2d = pca.fit_transform(Q)
K_2d = pca.fit_transform(K)
V_2d = pca.fit_transform(V)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(H_2d[:, 0], H_2d[:, 1], c=range(5), cmap='viridis')
axes[0, 0].set_title('Оригінальний простір H')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')

axes[0, 1].scatter(Q_2d[:, 0], Q_2d[:, 1], c=range(5), cmap='viridis')
axes[0, 1].set_title('Підпростір Query (Q)')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')

axes[1, 0].scatter(K_2d[:, 0], K_2d[:, 1], c=range(5), cmap='viridis')
axes[1, 0].set_title('Підпростір Key (K)')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')

axes[1, 1].scatter(V_2d[:, 0], V_2d[:, 1], c=range(5), cmap='viridis')
axes[1, 1].set_title('Підпростір Value (V)')
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('attention_projections.png', dpi=300)
print("\nВізуалізація збережена: attention_projections.png")
print("Це показує, як токени розподіляються в різних підпросторах Q, K, V")
```

**Ключові висновки:**
1. Self-Attention — це послідовність лінійних перетворень (тензорних операцій)
2. Q, K, V — це проекції у різні підпростори ознак, кожен з яких виділяє різні аспекти семантики
3. Attention weights визначають, як комбінувати інформацію з різних токенів
4. Геометрична інтерпретація допомагає зрозуміти, чому Self-Attention працює краще за прості методи

### Multi-Head Attention

**Ідея:** Використовуємо кілька "голів" attention паралельно, кожна з яких фокусується на різних аспектах (синтаксис, семантика, позиція).



**Формалізація:**

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}_O
$$

де:

$$
\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_Q^{(i)}, \mathbf{K}\mathbf{W}_K^{(i)}, \mathbf{V}\mathbf{W}_V^{(i)})
$$

**Параметри:**
- $h$ — кількість голів (зазвичай 8, 12, 16).
- $d_k = d_v = d / h$ — розмірність проекції кожної голови (дозволяє зберегти загальну обчислювальну складність).
- $\mathbf{W}_O \in \mathbb{R}^{hd_v \times d}$ — вихідна матриця, що проектує конкатенований вектор назад у простір розмірності $d$.

### Обчислювальна Складність Multi-Head Attention

**Критично важливо:** Розуміння складності алгоритму пояснює обмеження BERT та необхідність оптимізацій.

#### Аналіз складності крок за кроком



Розглянемо формулу Attention:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

де:
- $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{n \times d_k}$ — матриці Query, Key, Value
- $n$ — довжина послідовності (кількість токенів)
- $d_k$ — розмірність Key/Query/Value

**Крок 1: Обчислення $\mathbf{Q}\mathbf{K}^\top$**

- $\mathbf{Q}$ має розмірність $n \times d_k$
- $\mathbf{K}^\top$ має розмірність $d_k \times n$
- Результат (Attention Map): $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times n}$

**Складність матричного множення:**
$$O(n \cdot d_k \cdot n) = O(n^2 d_k)$$

Оскільки $d_k$ — константа архітектури (наприклад, 64 для BERT-base), складність по довжині послідовності: **$O(n^2)$**.

**Крок 2: Softmax над матрицею $n \times n$**

- Softmax обчислюється для кожного рядка матриці $\mathbf{Q}\mathbf{K}^\top$.
- Для кожного рядка: $n$ експонент + $n$ ділень.
- Всього рядків: $n$.
- **Складність:** $O(n^2)$.

**Крок 3: Множення на $\mathbf{V}$**

- Softmax результат: $n \times n$
- $\mathbf{V}$: $n \times d_k$
- Результат: $n \times d_k$

**Складність матричного множення:**
$$O(n \cdot n \cdot d_k) = O(n^2 d_k) \approx O(n^2)$$

**Загальна складність одного Attention head:**

$$
O(n^2) + O(n^2) + O(n^2) = O(n^2)
$$

**Для Multi-Head Attention з $h$ головами:**



Оскільки голови обчислюються паралельно (або послідовно, але незалежно), складність залишається квадратичною:

$$
\text{Multi-Head Attention: } O(h \cdot n^2) = O(n^2)
$$

(Оскільки $h$ — константа архітектури, наприклад, 12 для BERT-base).

#### Вплив на Пам'ять (Memory Complexity)

**Проблема:** Матриця attention weights має розмір $n \times n$.

**Приклад:**

- BERT-base: максимальна довжина $n = 512$ токенів
- Розмір матриці attention: $512 \times 512 = 262,144$ елементів
- Для 12 голів: $12 \times 262,144 = 3,145,728$ елементів
- Для float32 (4 байти): $3,145,728 \times 4 = 12.6$ МБ **лише для attention weights одного шару**

**Для 12 шарів BERT-base:** $12 \times 12.6 = 151.2$ МБ

**Для довших послідовностей:**

- $n = 1024$: $1024 \times 1024 = 1,048,576$ елементів на голову
- 12 голів × 12 шарів = 144 матриці
- Пам'ять: $144 \times 1,048,576 \times 4 = 603$ МБ

**Для $n = 2048$:** $2,415$ МБ ≈ **2.4 ГБ** лише для attention weights!

#### Чому BERT Має Обмеження на Довжину Контексту

**Технічні обмеження:**

1. **Пам'ять:** Квадратична залежність від $n$ → швидко вичерпується GPU пам'ять
2. **Час обчислення:** $O(n^2)$ → для $n = 2048$ в 16 разів повільніше, ніж для $n = 512$
3. **Архітектура:** BERT-base навчений на послідовностях до 512 токенів

**Практичні наслідки:**

```
Довжина послідовності (n) | Час обчислення (відносно) | Пам'ять attention
--------------------------|---------------------------|------------------
512 (BERT-base)           | 1×                         | 12.6 МБ/шар
1024                      | 4×                         | 50.4 МБ/шар
2048                      | 16×                        | 201.6 МБ/шар
4096                      | 64×                        | 806.4 МБ/шар
```

**Висновок:** Подвоєння довжини послідовності → **4-кратне збільшення** часу та пам'яті.

#### Завдання: Обчислення Складності

**Завдання 1:** Обчисліть складність для послідовності довжиною $n = 1024$ токенів (при $d_k = 64$).

**Рішення:**

**1. Обчислення $\mathbf{Q}\mathbf{K}^\top$:**
- Операцій: $n \cdot d_k \cdot n = 1024 \cdot 64 \cdot 1024 = 67,108,864$
- Складність: $O(1024^2) \approx O(10^6)$

**2. Softmax:**
- Для кожного з $n$ рядків: $n$ експонент + $n$ ділень.
- Операцій: $n \cdot 2n = 2n^2 = 2 \cdot 1,048,576 = 2,097,152$
- Складність: $O(n^2)$

**3. Множення на $\mathbf{V}$:**
- Операцій: $n \cdot n \cdot d_k = 1024 \cdot 1024 \cdot 64 = 67,108,864$
- Складність: $O(n^2)$

**Загальна складність:**

$$
O(n^2 d_k) \approx 6.7 \cdot 10^7 \text{ операцій для одного attention head.}
$$

**Для Multi-Head з $h = 12$:**

$$
O(h \cdot n^2 d_k) = 12 \cdot 67,108,864 \approx 805,306,368 \text{ операцій.}
$$

**Завдання 2:** Порівняйте складність для $n = 512$ та $n = 2048$.

**Рішення:**

- $n = 512$: $O(512^2) = O(262,144)$
- $n = 2048$: $O(2048^2) = O(4,194,304)$

**Відношення:** $\frac{2048^2}{512^2} = \frac{4,194,304}{262,144} = 16$

**Висновок:** Послідовність у 4 рази довша потребує **16 разів більше** обчислень.

#### Навіщо Потрібні Оптимізації: DistilBERT та Інші

**Проблема:** BERT-base має 110M параметрів та обмеження $n = 512$.

**Рішення 1: DistilBERT (Knowledge Distillation)**

**Ідея:** Навчити меншу модель, яка імітує поведінку великої.

**Параметри:**
- BERT-base: 110M параметрів, 12 шарів, 12 голів
- DistilBERT: 66M параметрів, 6 шарів, 12 голів

**Переваги:**
- **Швидкість:** ~2× швидше завдяки меншій кількості шарів
- **Пам'ять:** ~2× менше параметрів
- **Якість:** Зберігає ~97% точності BERT-base

**Математично:** Складність залишається $O(n^2)$, але константа менша через меншу кількість шарів.

**Рішення 2: Longformer (Linear Attention)**

**Ідея:** Замість повного attention ($n \times n$) використовувати локальний attention з вікном $w$.

**Складність:**
- Повний attention: $O(n^2)$
- Локальний attention: $O(n \times w)$, де $w$ — розмір вікна (константа)

**Приклад:** $w = 512$, $n = 4096$
- Повний: $O(4096^2) = O(16,777,216)$
- Локальний: $O(4096 \times 512) = O(2,097,152)$

**Прискорення:** $\frac{16,777,216}{2,097,152} = 8$ разів!

**Рішення 3: Sparse Attention (BigBird)**

**Ідея:** Використовувати sparse матрицю attention замість повної.

**Складність:** $O(n)$ замість $O(n^2)$ для певних паттернів.

#### Практичний Приклад: Порівняння Складності

```python
"""
Демонстрація складності Multi-Head Attention.
"""

import numpy as np
import time


def compute_attention_complexity(n: int, d_k: int = 64, h: int = 12) -> dict:
    """
    Обчислює складність attention для послідовності довжиною n.
    
    Returns:
        Словник з кількістю операцій та оцінкою часу
    """
    # Симуляція обчислень
    Q = np.random.randn(n, d_k)
    K = np.random.randn(n, d_k)
    V = np.random.randn(n, d_k)
    
    # Крок 1: QK^T
    start = time.time()
    QK_T = Q @ K.T  # n × d_k @ d_k × n = n × n
    ops_qkt = n * d_k * n
    time_qkt = time.time() - start
    
    # Крок 2: Softmax (спрощена версія)
    start = time.time()
    # Softmax для кожного рядка
    exp_scores = np.exp(QK_T / np.sqrt(d_k))
    softmax_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    ops_softmax = n * n * 2  # n експонент + n ділень для кожного рядка
    time_softmax = time.time() - start
    
    # Крок 3: Множення на V
    start = time.time()
    attention_output = softmax_scores @ V  # n × n @ n × d_k = n × d_k
    ops_v = n * n * d_k
    time_v = time.time() - start
    
    total_ops = ops_qkt + ops_softmax + ops_v
    total_time = time_qkt + time_softmax + time_v
    
    # Для h голів
    total_ops_multihead = total_ops * h
    total_time_multihead = total_time * h
    
    return {
        'n': n,
        'operations_qkt': ops_qkt,
        'operations_softmax': ops_softmax,
        'operations_v': ops_v,
        'total_operations': total_ops,
        'total_operations_multihead': total_ops_multihead,
        'time_seconds': total_time_multihead,
        'complexity_order': f'O(n²) = O({n}²) = O({n*n})'
    }


def compare_sequence_lengths():
    """
    Порівнює складність для різних довжин послідовностей.
    """
    print("=" * 80)
    print("АНАЛІЗ СКЛАДНОСТІ MULTI-HEAD ATTENTION")
    print("=" * 80)
    print()
    
    sequence_lengths = [128, 256, 512, 1024, 2048]
    results = []
    
    for n in sequence_lengths:
        result = compute_attention_complexity(n)
        results.append(result)
    
    print(f"{'n':<8} {'Операції (QK^T)':<20} {'Операції (Softmax)':<20} {'Операції (V)':<20} {'Всього (×12)':<20} {'Час (сек)':<12}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['n']:<8} {r['operations_qkt']:<20,} {r['operations_softmax']:<20,} {r['operations_v']:<20,} {r['total_operations_multihead']:<20,} {r['time_seconds']:<12.4f}")
    
    print()
    print("ВИСНОВКИ:")
    print("-" * 80)
    
    # Порівняння відносно n=512
    base_n = 512
    base_result = next(r for r in results if r['n'] == base_n)
    base_ops = base_result['total_operations_multihead']
    
    print(f"Базовий випадок: n = {base_n}, операцій = {base_ops:,}")
    print()
    
    for r in results:
        if r['n'] != base_n:
            ratio = r['total_operations_multihead'] / base_ops
            n_ratio = r['n'] / base_n
            print(f"n = {r['n']:<6} ({n_ratio:.1f}× довше): {r['total_operations_multihead']:>20,} операцій ({ratio:>6.2f}× більше)")
            print(f"  Очікуване співвідношення: ({n_ratio:.1f})² = {n_ratio**2:.2f}×")
            print()
    
    print("=" * 80)
    print("КЛЮЧОВІ ВИСНОВКИ:")
    print("=" * 80)
    print("1. Складність Multi-Head Attention: O(n²)")
    print("2. Подвоєння довжини послідовності → 4-кратне збільшення операцій")
    print("3. Це пояснює обмеження BERT на довжину контексту (512 токенів)")
    print("4. Для довших послідовностей потрібні оптимізації (Longformer, BigBird)")
    print("=" * 80)


if __name__ == "__main__":
    compare_sequence_lengths()
```

**Очікуваний вивід:**

```
================================================================================
АНАЛІЗ СКЛАДНОСТІ MULTI-HEAD ATTENTION
================================================================================

n        Операції (QK^T)     Операції (Softmax)  Операції (V)         Всього (×12)          Час (сек)   
----------------------------------------------------------------------------------------------------
128      1,048,576           32,768              1,048,576            25,477,632            0.0123
256      4,194,304           131,072             4,194,304            102,341,376           0.0456
512      16,777,216          524,288             16,777,216           409,364,736           0.1824
1024     67,108,864          2,097,152           67,108,864           1,637,458,944          0.7296
2048     268,435,456         8,388,608           268,435,456          6,549,835,776          2.9184

ВИСНОВКИ:
--------------------------------------------------------------------------------
Базовий випадок: n = 512, операцій = 409,364,736

n = 128   (0.3× довше):        25,477,632 операцій ( 0.06× більше)
  Очікуване співвідношення: (0.3)² = 0.09×

n = 256   (0.5× довше):       102,341,376 операцій ( 0.25× більше)
  Очікуване співвідношення: (0.5)² = 0.25×

n = 1024  (2.0× довше):     1,637,458,944 операцій ( 4.00× більше)
  Очікуване співвідношення: (2.0)² = 4.00×

n = 2048  (4.0× довше):     6,549,835,776 операцій (16.00× більше)
  Очікуване співвідношення: (4.0)² = 16.00×
```

#### Ключові Висновки про Складність

1. **Квадратична залежність:** Складність Multi-Head Attention є $O(n^2)$, що означає швидке зростання обчислень з довжиною послідовності.

2. **Обмеження BERT:** Максимальна довжина 512 токенів обумовлена:
   - Обмеженням пам'яті GPU
   - Часом обчислення
   - Архітектурою навчання

3. **Необхідність оптимізацій:**
   - **DistilBERT:** Зменшує кількість шарів (константу), але складність залишається $O(n^2)$
   - **Longformer/BigBird:** Змінюють саму складність на $O(n)$ або $O(n \log n)$

4. **Практичні наслідки:** Для технічних логів, які можуть бути довгими, важливо:
   - Обрізати або сегментувати довгі послідовності
   - Використовувати оптимізовані моделі (Longformer для довгих документів)
   - Розглянути альтернативні архітектури для production

## BERT: Bidirectional Encoder Representations

### Ключова Ідея

**Word2Vec та ELMo:** Односторонні (left-to-right або right-to-left)

**BERT:** Двобічний (bidirectional) — одночасно бачить весь контекст з обох сторін.

### Архітектура BERT

**Base BERT:**
- 12 encoder layers
- 12 attention heads
- 768 hidden dimensions
- 110M parameters

**Large BERT:**
- 24 encoder layers
- 16 attention heads
- 1024 hidden dimensions
- 340M parameters

### Pre-training Задачі

**1. Masked Language Model (MLM):**

Випадково маскуємо 15% токенів та навчаємо модель передбачати їх.

**Приклад:**

```
Input:  "Connection [MASK] by database"
Target: "Connection refused by database"
```

**2. Next Sentence Prediction (NSP):**

Навчаємо модель розуміти зв'язок між реченнями.

**Приклад:**

```
Sentence A: "Connection refused"
Sentence B: "Database unavailable"
Label: IsNext (чи B слідує за A?)
```

### Спеціальні Токени

**`[CLS]`** — Classification token:
- Розміщується на початку кожного речення
- Використовується для класифікації
- Кодує загальне представлення всього речення

**`[SEP]`** — Separator token:
- Розділяє пари речень
- Використовується в NSP

**`[MASK]`** — Masked token:
- Використовується в MLM

## Використання `[CLS]` для Класифікації

### Ідея

Після обробки через BERT, токен `[CLS]` містить контекстне представлення всього речення.



**Формалізація:**

$$
\mathbf{h}_{\texttt{[CLS]}} = \text{BERT}(\texttt{"Connection refused by database"})[0]
$$

де індекс $[0]$ позначає позицію спеціального токена `[CLS]`.

**Класифікація (Logistic Regression шар зверху BERT):**

$$
\mathbf{y} = \text{softmax}(\mathbf{W} \mathbf{h}_{\texttt{[CLS]}} + \mathbf{b})
$$

де:
- $\mathbf{W} \in \mathbb{R}^{C \times d}$ — матриця ваг класифікатора.
- $C$ — кількість класів.
- $d$ — розмірність прихованого стану (768 для BERT-base).

### Чому Це Працює Краще за Naive Bayes

**Приклад:**

```
Лог 1: "Connection refused by database"
Лог 2: "Connection established successfully"
```

**Naive Bayes:**
- Обидва містять "Connection"
- "Connection" частіше в Normal → обидва класифікуються як Normal

**BERT:**
- `[CLS]` токен враховує весь контекст
- "refused" vs "established" змінюють значення `[CLS]`
- Правильна класифікація: Critical vs Normal

## Explainability: Чому BERT Вирішив, Що Цей Лог — Critical?

**Для інженерів критично важливо:** Коли BERT класифікує лог як "Critical", інженер має розуміти, **чому** модель прийняла таке рішення. Це дозволяє:
- Перевірити правильність класифікації
- Виявити помилки моделі
- Покращити модель на основі аналізу помилок
- Збільшити довіру до системи моніторингу

### Механізм Візуалізації Attention Weights

**Ідея:** Attention weights показують, на які слова модель "звертає увагу" при прийнятті рішення. Високі attention weights означають, що ці слова "активували" тривогу.

**Математична формалізація (Interpretability):**

Для класифікації через `[CLS]` токен, ми можемо проаналізувати attention weights між `[CLS]` та іншими токенами (які слова модель "слухала", приймаючи рішення):



$$
\alpha_{\texttt{[CLS]}, j} = \frac{\exp(\text{score}_{\texttt{[CLS]}, j})}{\sum_{k=1}^{n} \exp(\text{score}_{\texttt{[CLS]}, k})}
$$

де:

$$
\text{score}_{\texttt{[CLS]}, j} = \frac{\mathbf{q}_{\texttt{[CLS]}} \cdot \mathbf{k}_j}{\sqrt{d_k}}
$$

**Інтерпретація:**
- **Високий $\alpha_{\texttt{[CLS]}, j}$:** Токен $j$ сильно впливає на формування вектора $\mathbf{h}_{\texttt{[CLS]}}$, а отже — на фінальну класифікацію.
- **Низький $\alpha_{\texttt{[CLS]}, j}$:** Токен $j$ майже ігнорується моделлю (шум або нерелевантна інформація).

### Практичний Приклад: Візуалізація Attention для Логів

**Приклад 1: "Connection refused by database server"**

```python
"""
Візуалізація attention weights для пояснення класифікації.
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

def visualize_attention(text: str, model, tokenizer, layer_idx: int = 11):
    """
    Візуалізує attention weights для тексту.
    
    Args:
        text: Текст для аналізу
        model: BERT модель
        tokenizer: BERT токенізатор
        layer_idx: Індекс шару для аналізу (останній шар зазвичай найінформативніший)
    """
    # Токенізація
    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    # Отримуємо attention weights
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)
        attentions = outputs.attentions  # Список attention weights для кожного шару
    
    # Вибираємо останній шар та першу голову (для простоти)
    attention_layer = attentions[layer_idx]  # (batch, heads, seq_len, seq_len)
    attention_head_0 = attention_layer[0, 0, :, :].cpu().numpy()  # Перша голова
    
    # Фокусуємося на attention від [CLS] до інших токенів
    cls_attention = attention_head_0[0, :]  # Attention від [CLS] до всіх токенів
    
    # Візуалізація
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Створюємо heatmap
    sns.heatmap(
        attention_head_0,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        annot=False,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_title(f'Attention Weights (Layer {layer_idx}, Head 0)\nText: "{text}"', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Tokens', fontsize=12)
    ax.set_ylabel('Query Tokens', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'attention_visualization_{text[:20].replace(" ", "_")}.png', dpi=300)
    plt.show()
    
    # Виводимо топ-5 токенів, на які [CLS] звертає найбільшу увагу
    top_indices = cls_attention.argsort()[-5:][::-1]
    print("\nТоп-5 токенів, які найбільше впливають на класифікацію:")
    print("-" * 60)
    for idx in top_indices:
        print(f"  {tokens[idx]:<20} Attention: {cls_attention[idx]:.4f}")
    
    return cls_attention, tokens

# Приклад використання
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

text = "Connection refused by database server"
cls_attention, tokens = visualize_attention(text, model, tokenizer)
```

**Очікуваний результат:**

```
Топ-5 токенів, які найбільше впливають на класифікацію:
------------------------------------------------------------
  refused                Attention: 0.3245
  database               Attention: 0.2156
  Connection             Attention: 0.1892
  server                 Attention: 0.1123
  by                     Attention: 0.0876
```

**Інтерпретація для інженера:**
- Модель звертає найбільшу увагу на слово "refused" — це ключовий індикатор проблеми
- "database" та "Connection" також важливі — показують об'єкт та тип проблеми
- Слово "by" має низьку увагу — це службове слово, не важливе для класифікації

### Multi-Head Attention: Різні Аспекти Уваги

**Важливо:** BERT використовує Multi-Head Attention (12 голів для BERT-base). Кожна голова може фокусуватися на різних аспектах:

- **Голова 1:** Може фокусуватися на синтаксичних зв'язках (наприклад, "refused" → "Connection")
- **Голова 2:** Може фокусуватися на семантичних зв'язках (наприклад, "database" → "server")
- **Голова 3:** Може фокусуватися на позиційних зв'язках (наприклад, порядок слів)

**Візуалізація всіх голів:**

```python
def visualize_all_heads(text: str, model, tokenizer, layer_idx: int = 11):
    """
    Візуалізує attention weights для всіх голів.
    """
    encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True)
        attentions = outputs.attentions
    
    attention_layer = attentions[layer_idx]  # (batch, heads, seq_len, seq_len)
    num_heads = attention_layer.shape[1]
    
    # Створюємо grid для візуалізації всіх голів
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        attention_head = attention_layer[0, head_idx, :, :].cpu().numpy()
        cls_attention = attention_head[0, :]
        
        ax = axes[head_idx]
        ax.barh(range(len(tokens)), cls_attention)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel('Attention Weight')
        ax.set_title(f'Head {head_idx}')
        ax.invert_yaxis()
    
    plt.suptitle(f'Attention Weights for All Heads (Layer {layer_idx})\nText: "{text}"', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'attention_all_heads_{text[:20].replace(" ", "_")}.png', dpi=300)
    plt.show()
```

### Практичне Застосування для AI-SRE

**Сценарій:** Інженер отримує сповіщення "Critical" для логу "Connection established successfully".

**Крок 1: Перевірка класифікації**
- Візуалізуємо attention weights
- Бачимо, що модель звертає увагу на "Connection" та "established"
- Але "successfully" також має високу увагу — це може бути помилка

**Крок 2: Аналіз помилки**
- Модель неправильно інтерпретувала контекст
- Можливо, навчальні дані містили помилки
- Потрібно додати більше прикладів з "established successfully" як Normal

**Крок 3: Покращення моделі**
- Додаємо більше прикладів у навчальний набір
- Fine-tune модель на виправлених даних
- Перевіряємо attention weights після переобучення

### Інструменти для Візуалізації

**1. Transformers Interpret (HuggingFace):**
```python
from transformers_interpret import SequenceClassificationExplainer

explainer = SequenceClassificationExplainer(model, tokenizer)
word_attributions = explainer("Connection refused by database server")
explainer.visualize()
```

**2. BertViz:**
```python
from bertviz import head_view

# Інтерактивна візуалізація attention weights
head_view(attention_weights, tokens)
```

**3. Captum (PyTorch):**
```python
from captum.attr import IntegratedGradients

# Атрибуція важливості токенів через Integrated Gradients
ig = IntegratedGradients(model)
attributions = ig.attribute(inputs, target=predicted_class)
```

### Ключові Висновки про Explainability

1. **Attention weights показують важливість токенів:** Високі attention weights означають, що токен сильно впливає на класифікацію
2. **Multi-Head Attention дає багатоаспектний погляд:** Різні голови фокусуються на різних аспектах семантики
3. **Візуалізація допомагає виявити помилки:** Інженер може перевірити, чи модель правильно інтерпретує контекст
4. **Explainability критична для production:** Довіра до системи залежить від здатності пояснити рішення

**Для інженерів:** Коли BERT класифікує лог як "Critical", завжди перевіряйте attention weights, щоб зрозуміти, які слова "активували" тривогу. Це допоможе виявити помилки моделі та покращити систему моніторингу.

## Від Теореми Байєса до Крос-Ентропії: Математичний Зв'язок

### Інформаційна ентропія

**Визначення:** Ентропія Шеннона вимірює рівень хаосу або невизначеності випадкової величини.



$$
H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)
$$

де $X$ — випадкова величина з можливими значеннями $\{x_1, x_2, \ldots, x_n\}$.

**Інтерпретація:**
- $H(X) = 0$ — повна визначеність (одна подія трапляється з ймовірністю 1).
- $H(X) \to \max$ — максимальна невизначеність (всі події рівноймовірні, наприклад, підкидання чесної монети).

**Приклад для класифікації:**

Для бінарної класифікації (`Normal`/`Critical`):

$$
H(Y) = -P(\text{Normal}) \log_2 P(\text{Normal}) - P(\text{Critical}) \log_2 P(\text{Critical})
$$

Якщо $P(\text{Normal}) = 0.99$, $P(\text{Critical}) = 0.01$:

$$
H(Y) = -0.99 \log_2 0.99 - 0.01 \log_2 0.01 \approx 0.081
$$

**Висновок:** Низька ентропія — висока **визначеність** (система впевнена, що це клас `Normal`).

### Крос-ентропія (Cross-Entropy)

**Визначення:** Крос-ентропія вимірює розбіжність між істинним розподілом $P$ та прогнозованим розподілом $Q$. Це основна функція втрат (Loss Function) для класифікації.

$$
H(P, Q) = -\sum_{i=1}^{n} P(x_i) \log Q(x_i)
$$

**Для задачі класифікації (Categorical Cross-Entropy):**

$$
H(y, \hat{y}) = -\sum_{c=1}^{C} y_c \log \hat{y}_c
$$

де:
- $y_c$ — істинна ймовірність класу $c$ (використовується **One-hot encoding**: 1 для правильного класу, 0 для інших).
- $\hat{y}_c$ — передбачена моделлю ймовірність класу $c$ (вихід Softmax).

**Приклад:**

Для логу `Connection refused` з істинним класом **Critical**:

$$
\mathbf{y} = [0, 1] \quad \text{(Normal=0, Critical=1)}
$$



**Сценарій 1: Модель впевнена (Правильне передбачення)**

$$
\hat{\mathbf{y}} = [0.1, 0.9] \quad \text{(модель каже: 90\% Critical)}
$$

$$
H(\mathbf{y}, \hat{\mathbf{y}}) = -0 \cdot \ln 0.1 - 1 \cdot \ln 0.9 = -\ln 0.9 \approx 0.105
$$

**Сценарій 2: Модель помиляється (Фатальна помилка)**

$$
\hat{\mathbf{y}} = [0.9, 0.1] \quad \text{(модель каже: 90\% Normal)}
$$

$$
H(\mathbf{y}, \hat{\mathbf{y}}) = -0 \cdot \ln 0.9 - 1 \cdot \ln 0.1 = -\ln 0.1 \approx 2.303
$$

**Висновок:** Функція Cross-Entropy жорстоко штрафує за впевнені неправильні передбачення (Loss зростає експоненціально, коли $\hat{y} \to 0$ для правильного класу).

### Зв'язок з Теоремою Байєса

**Теорема Байєса:**

$$
P(c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c) \cdot P(c)}{P(\mathbf{x})}
$$

**Максимізація апостеріорної ймовірності (MAP):**

$$
\hat{c} = \operatorname*{arg\,max}_{c} P(c \mid \mathbf{x}) = \operatorname*{arg\,max}_{c} \log P(c \mid \mathbf{x})
$$

*(Логарифмування — монотонна функція, тому точка максимуму не змінюється, але добуток ймовірностей перетворюється на суму логарифмів, що є обчислювально стабільнішим).*



**Еквівалентність мінімізації крос-ентропії:**

Максимізація $\log P(c \mid \mathbf{x})$ еквівалентна мінімізації $-\log P(c \mid \mathbf{x})$.

**Формалізація для набору даних:**

Нехай маємо навчальний набір $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$.

**1. Максимізація лог-правдоподібності (MLE):**

$$
\max_{\theta} \sum_{i=1}^{N} \log P(y_i \mid x_i; \theta)
$$

**2. Еквівалентна мінімізація негативної лог-правдоподібності (NLL):**

$$
\min_{\theta} \left( -\sum_{i=1}^{N} \log P(y_i \mid x_i; \theta) \right)
$$

**3. Це є сумою крос-ентропій:**

Оскільки для One-hot вектора $y_i$ лише один елемент дорівнює 1, а інші 0:

$$
\min_{\theta} \sum_{i=1}^{N} H(y_i, \hat{y}_i) = \min_{\theta} \sum_{i=1}^{N} \left( -\sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c} \right)
$$

де $\hat{y}_{i,c} = P(c \mid x_i; \theta)$ — передбачення моделі (вихід Softmax).

**Висновок:** Навчання нейромережі через мінімізацію Cross-Entropy Loss є прямою реалізацією принципу максимальної правдоподібності (MLE).

### Від Байєса до BERT

**Naive Bayes (ймовірнісний підхід):**

1. Оцінюємо $P(c | \mathbf{x})$ через теорему Байєса
2. Максимізуємо апостеріорну ймовірність
3. Еквівалентно мінімізації крос-ентропії (якщо використовувати лог-ймовірності)

**BERT (нейронна мережа):**

1. Навчаємо параметри $\theta$ (ваги мережі)
2. Мінімізуємо крос-ентропію як функцію втрат
3. Еквівалентно максимізації лог-ймовірності (через backpropagation)

**Математична еквівалентність:**

$$
\text{Naive Bayes: } \hat{c} = \operatorname*{arg\,max}_{c} P(c \mid \mathbf{x}) = \operatorname*{arg\,max}_{c} \left( \log P(\mathbf{x} \mid c) + \log P(c) \right)
$$

$$
\text{BERT: } \hat{c} = \operatorname*{arg\,max}_{c} \hat{y}_c = \operatorname*{arg\,max}_{c} \left( \text{softmax}(\mathbf{W} \mathbf{h}_{\texttt{[CLS]}} + \mathbf{b}) \right)_c
$$

Обидва підходи мінімізують неузгодженість (крос-ентропію), але:
- **Naive Bayes:** Параметри — це ймовірності $P(w \mid c)$, які оцінюються статистично (частотний аналіз).
- **BERT:** Параметри — це ваги $\theta$, які оцінюються ітеративно (градієнтний спуск).

### Крос-ентропія як функція втрат у BERT

**Формалізація:**

Для класифікації з $C$ класами мінімізуємо функцію втрат:

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}(\theta)
$$

де:
- $N$ — розмір навчального набору.
- $y_{i,c} \in \{0, 1\}$ — one-hot encoding істинного класу.
- $\hat{y}_{i,c}(\theta) = \text{softmax}(\mathbf{W} \mathbf{h}_{\texttt{[CLS]}}^{(i)} + \mathbf{b})_c$ — передбачена ймовірність.



**Градієнт (Backpropagation):**

$$
\frac{\partial \mathcal{L}}{\partial \theta} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \frac{y_{i,c}}{\hat{y}_{i,c}} \frac{\partial \hat{y}_{i,c}}{\partial \theta}
$$

*(Примітка: При поєднанні Softmax + Cross-Entropy похідна суттєво спрощується до різниці $\hat{y} - y$, що робить навчання стабільним).*

**Backpropagation:** Градієнт поширюється назад через мережу для оновлення параметрів.

### Чому Крос-Ентропія?

**Переваги:**

1. **Математична обґрунтованість:** Зв'язана з максимізацією лог-ймовірності
2. **Чутливість до помилок:** Великі штрафи за великі помилки (через логарифм)
3. **Чисельна стабільність:** Працює добре з softmax
4. **Інтерпретація:** Мінімізація крос-ентропії = максимізація ймовірності правильних передбачень

**Порівняння з іншими функціями втрат:**

### Альтернатива: Mean Squared Error (MSE)

Зазвичай використовується для задач регресії (наприклад, передбачення ціни або similarity score), а не класифікації.

**Формалізація:**

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \| \mathbf{y}_i - \hat{\mathbf{y}}_i \|^2
$$

**Чому MSE рідко використовують для класифікації з Softmax?**

1.  **Проблема зникаючих градієнтів (Vanishing Gradients):**
    У поєднанні з Softmax/Sigmoid, якщо модель впевнено помиляється (наприклад, $\hat{y} \approx 0$ при $y=1$), похідна функції втрат MSE стає близькою до нуля. Навчання в цій точці фактично зупиняється ("насичення" нейронів). Cross-Entropy компенсує це, маючи крутий градієнт при великих помилках.

2.  **Неопуклість (Non-convexity):**
    
    Поверхня помилки MSE для нейромереж класифікації часто є неопуклою (має багато локальних мінімумів), що ускладнює роботу градієнтного спуску. Cross-Entropy гарантує опуклішу поверхню для логістичної регресії та простішу для глибоких мереж.

**Порівняння штрафів:**

- **Cross-Entropy:** Штрафує експоненціально ($-\ln \hat{y}$).
    * Якщо $\hat{y}_{\text{true}} = 0.0001$, Loss $\to \infty$. Модель змушена виправити грубу помилку негайно.
- **MSE:** Штрафує квадратично ($(1 - \hat{y})^2$).
    * Максимальна помилка обмежена одиницею ($1^2 = 1$). Модель може "ігнорувати" складні приклади, оскільки штраф не є катастрофічним.

**Висновок:** MSE погано підходить для класифікації, оскільки не враховує ймовірнісну природу задачі (дискретність класів). Крос-ентропія (MLE) є математично обґрунтованим вибором для розподілів Бернуллі/Мультиноміальних.


### Практичний Приклад

```python
"""
Демонстрація зв'язку між теоремою Байєса та крос-ентропією.
"""

import torch
import torch.nn as nn
import numpy as np


def demonstrate_cross_entropy():
    """
    Демонструє, як крос-ентропія пов'язана з максимізацією ймовірності.
    """
    print("=" * 70)
    print("ВІД ТЕОРЕМИ БАЙЄСА ДО КРОС-ЕНТРОПІЇ")
    print("=" * 70)
    print()
    
    # Істинні мітки (one-hot encoding)
    y_true = torch.tensor([[0, 1], [1, 0], [0, 1]])  # Critical, Normal, Critical
    
    # Передбачення моделі (ймовірності)
    y_pred_good = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]])  # Добре передбачення
    y_pred_bad = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])   # Погане передбачення
    
    # Крос-ентропія
    criterion = nn.CrossEntropyLoss()
    
    # Конвертуємо в формат для CrossEntropyLoss (індекси класів)
    y_true_indices = torch.tensor([1, 0, 1])  # Critical=1, Normal=0
    
    # Для CrossEntropyLoss потрібні logits (до softmax)
    logits_good = torch.log(y_pred_good / (1 - y_pred_good.sum(dim=1, keepdim=True) + y_pred_good))
    logits_bad = torch.log(y_pred_bad / (1 - y_pred_bad.sum(dim=1, keepdim=True) + y_pred_bad))
    
    # Або використовуємо BCEWithLogitsLoss для бінарної класифікації
    bce_loss = nn.BCELoss()
    
    loss_good = bce_loss(y_pred_good, y_true.float())
    loss_bad = bce_loss(y_pred_bad, y_true.float())
    
    print("Добре передбачення:")
    print(f"  Передбачення: {y_pred_good.numpy()}")
    print(f"  Крос-ентропія: {loss_good.item():.4f}")
    print()
    
    print("Погане передбачення:")
    print(f"  Передбачення: {y_pred_bad.numpy()}")
    print(f"  Крос-ентропія: {loss_bad.item():.4f}")
    print()
    
    print("Висновок:")
    print("  Крос-ентропія більша для поганих передбачень.")
    print("  Мінімізація крос-ентропії = максимізація ймовірності правильних передбачень.")
    print()
    
    # Демонстрація зв'язку з лог-ймовірністю
    print("Зв'язок з лог-ймовірністю:")
    print("-" * 70)
    
    # Для одного прикладу
    y_true_single = torch.tensor([0, 1])  # Critical
    y_pred_single = torch.tensor([0.1, 0.9])  # Передбачення: Critical з ймовірністю 0.9
    
    # Лог-ймовірність правильного класу
    log_prob = torch.log(y_pred_single[1])  # log P(Critical | x)
    print(f"  Лог-ймовірність правильного класу: {log_prob.item():.4f}")
    
    # Крос-ентропія
    ce_loss = -torch.sum(y_true_single * torch.log(y_pred_single))
    print(f"  Крос-ентропія: {ce_loss.item():.4f}")
    print()
    print("  Висновок: Крос-ентропія = -лог-ймовірність правильного класу")
    print("  Мінімізація крос-ентропії = максимізація лог-ймовірності")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_cross_entropy()
```

### Ключові Висновки

1. **Ентропія вимірює невизначеність:** Низька ентропія = висока впевненість у передбаченні.

2. **Крос-ентропія мірить помилку:** Відстань між істинним та передбаченим розподілом.

3. **Зв'язок з Байєсом:** Мінімізація крос-ентропії еквівалентна максимізації лог-ймовірності (Maximum Likelihood).

4. **BERT використовує крос-ентропію:** Як функцію втрат для навчання через градієнтний спуск.

5. **Математична еквівалентність:** Naive Bayes (підрахунок) та BERT (градієнтний спуск) обидва мінімізують крос-ентропію, але різними методами.

## Порівняння з Попередніми Методами

### Bag of Words

**Проблема:** Втрачає порядок та контекст.

```
"Connection refused" ≠ "Refused connection"
```

**BERT:** Розрізняє порядок через positional encoding.

### Naive Bayes

**Проблема:** Припущення незалежності слів.

```
P("Connection", "refused" | Critical) ≠ P("Connection" | Critical) × P("refused" | Critical)
```

**BERT:** Self-attention явно моделює залежності між словами.

### Word2Vec

**Проблема:** Статичні embeddings, полісемія.

```
"bank" (ріка) = "bank" (фінанси)
```

**BERT:** Контекстні embeddings, різні вектори для різних значень.

## Реалізація: BERT для Класифікації Логів

```python
"""
Використання BERT для класифікації технічних логів.
Демонструє переваги контекстних embeddings над статистичними методами.
"""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, 
    BertModel, 
    BertForSequenceClassification
)
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Результат класифікації BERT."""
    predicted_class: str
    probabilities: Dict[str, float]
    cls_embedding: torch.Tensor


class BERTLogClassifier:
    """
    Класифікатор логів на основі BERT.
    
    Використовує [CLS] токен для отримання представлення всього речення
    та класифікує його через лінійний шар.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        max_length: int = 128
    ):
        """
        Args:
            model_name: Назва попередньо навченої BERT моделі
            num_classes: Кількість класів (наприклад, Critical vs Normal)
            max_length: Максимальна довжина послідовності
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Завантажуємо токенізатор та модель
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        
        # Переводимо в режим оцінки
        self.model.eval()
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Токенізує текст для BERT.
        
        Returns:
            Словник з 'input_ids', 'attention_mask', 'token_type_ids'
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded
    
    def get_cls_embedding(self, text: str) -> torch.Tensor:
        """
        Отримує embedding [CLS] токена для тексту.
        
        Це представлення всього речення після обробки через BERT.
        """
        encoded = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.model.bert(**encoded)
            # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_dim]
            # [CLS] токен завжди на позиції 0
            cls_embedding = outputs.last_hidden_state[0, 0, :]
        
        return cls_embedding
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Класифікує текст.
        
        Returns:
            ClassificationResult з передбаченим класом та ймовірностями
        """
        encoded = self.tokenize(text)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Отримуємо передбачений клас
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        predicted_class = f"class_{predicted_class_id}"
        
        # Конвертуємо ймовірності в словник
        prob_dict = {
            f"class_{i}": probabilities[0, i].item()
            for i in range(self.num_classes)
        }
        
        # Отримуємо [CLS] embedding для аналізу
        cls_embedding = self.get_cls_embedding(text)
        
        return ClassificationResult(
            predicted_class=predicted_class,
            probabilities=prob_dict,
            cls_embedding=cls_embedding
        )
    
    def compare_contexts(self, texts: List[str]) -> Dict[str, float]:
        """
        Порівнює контекстні embeddings для різних текстів.
        
        Демонструє, як BERT розрізняє контексти навіть з однаковими словами.
        """
        embeddings = [self.get_cls_embedding(text) for text in texts]
        
        # Обчислюємо косинусну схожість між embeddings
        similarities = {}
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                emb1 = embeddings[i]
                emb2 = embeddings[j]
                
                # Косинусна схожість
                cosine_sim = torch.nn.functional.cosine_similarity(
                    emb1.unsqueeze(0), 
                    emb2.unsqueeze(0)
                ).item()
                
                key = f"'{texts[i]}' vs '{texts[j]}'"
                similarities[key] = cosine_sim
        
        return similarities


def demonstrate_bert_vs_naive_bayes() -> None:
    """
    Демонструє переваги BERT над Naive Bayes для контекстно-залежних логів.
    """
    print("=" * 70)
    print("BERT vs NAIVE BAYES: КОНТЕКСТНА ЗАЛЕЖНІСТЬ")
    print("=" * 70)
    print()
    
    # Приклади логів, де контекст критичний
    test_logs = [
        "Connection refused by database",
        "Connection established successfully",
        "Database connection timeout",
        "Database connection ready"
    ]
    
    print("ТЕСТОВІ ЛОГИ:")
    for i, log in enumerate(test_logs, 1):
        print(f"  {i}. {log}")
    print()
    
    # Створюємо класифікатор
    # Примітка: Для повної демонстрації потрібно fine-tune на даних
    # Тут показуємо лише отримання embeddings
    classifier = BERTLogClassifier(num_classes=2)
    
    print("BERT [CLS] EMBEDDINGS:")
    print("-" * 70)
    for log in test_logs:
        cls_emb = classifier.get_cls_embedding(log)
        print(f"  '{log}':")
        print(f"    Embedding shape: {cls_emb.shape}")
        print(f"    Embedding norm: {cls_emb.norm().item():.4f}")
    print()
    
    # Порівняння контекстів
    print("ПОРІВНЯННЯ КОНТЕКСТІВ (косинусна схожість):")
    print("-" * 70)
    
    # Групуємо логі за типом
    critical_logs = [
        "Connection refused by database",
        "Database connection timeout"
    ]
    
    normal_logs = [
        "Connection established successfully",
        "Database connection ready"
    ]
    
    # Порівнюємо всередині груп та між групами
    all_logs = critical_logs + normal_logs
    similarities = classifier.compare_contexts(all_logs)
    
    for pair, similarity in similarities.items():
        print(f"  {pair}: {similarity:.4f}")
    print()
    
    print("=" * 70)
    print("ВИСНОВОК:")
    print("  BERT розрізняє контексти навіть з однаковими словами.")
    print("  'Connection refused' та 'Connection established' мають")
    print("  різні [CLS] embeddings, що дозволяє правильно класифікувати.")
    print()
    print("  Naive Bayes не може розрізнити ці контексти,")
    print("  бо припускає незалежність слів.")
    print("=" * 70)


def demonstrate_attention_visualization() -> None:
    """
    Демонструє, як можна візуалізувати attention weights.
    Показує, на які слова модель "звертає увагу".
    """
    print("\n" + "=" * 70)
    print("ВІЗУАЛІЗАЦІЯ ATTENTION WEIGHTS")
    print("=" * 70)
    print()
    
    text = "Connection refused by database server"
    classifier = BERTLogClassifier()
    encoded = classifier.tokenize(text)
    
    # Отримуємо токени для візуалізації
    tokens = classifier.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
    
    print(f"Текст: '{text}'")
    print(f"Токени: {tokens[:10]}...")  # Показуємо перші 10
    print()
    print("Примітка: Для повної візуалізації attention weights")
    print("потрібно модифікувати модель для повернення attention.")
    print("Це можна зробити через model.config.output_attentions=True")
    print()


if __name__ == "__main__":
    # Примітка: Цей код потребує встановлення transformers та torch
    # pip install transformers torch
    
    try:
        demonstrate_bert_vs_naive_bayes()
        demonstrate_attention_visualization()
    except ImportError:
        print("Помилка: Потрібно встановити transformers та torch")
        print("  pip install transformers torch")
    except Exception as e:
        print(f"Помилка: {e}")
        print("Примітка: Для повної роботи потрібен fine-tuned BERT")
        print("на технічних логах. Це демонстрація концепції.")
```

## Чому BERT Краще для Технічних Логів

### 1. Контекстна Залежність

**Приклад:**

```
"Connection refused" → Critical
"Connection established" → Normal
```

BERT розрізняє через Self-Attention: "refused" змінює значення "Connection".

### 2. Послідовність Подій

**Приклад:**

```
Лог 1: "Error: timeout"
Лог 2: "Error: timeout Error: timeout Error: timeout" (×100)
```

BERT бачить частоту через positional encoding та attention. Одна помилка ≠ 100 помилок.

### 3. Аномалії в Послідовності

**Приклад:**

```
Нормальна послідовність: "Request → Process → Response"
Аномальна: "Request → Error → Timeout → Crash"
```

BERT моделює залежності між подіями через багатошаровий encoder.

### 4. Полісемія

**Приклад:**

```
"Service down" (сервіс не працює)
"Service up" (сервіс працює)
```

BERT розрізняє значення "down" та "up" через контекст.

## Обмеження BERT

### 1. Обчислювальна Складність

**Self-Attention:** $O(n^2)$ де $n$ — довжина послідовності.

**Детальний аналіз:** Див. розділ "Обчислювальна Складність Multi-Head Attention" вище, де розглянуто:
- Математичне обґрунтування $O(n^2)$ складності
- Вплив на пам'ять (квадратична залежність)
- Практичні наслідки для різних довжин послідовностей
- Завдання на обчислення складності

**Для довгих логів:** Потрібні оптимізації:
- **Longformer:** Локальний attention зі складністю $O(n \times w)$ замість $O(n^2)$
- **BigBird:** Sparse attention зі складністю $O(n)$ для певних паттернів
- **Reformer:** Locality-sensitive hashing для зменшення складності

### 2. Потребує Fine-tuning

BERT попередньо навчений на загальних текстах (Wikipedia, BooksCorpus). Для технічних логів потрібен fine-tuning на доменних даних, щоб модель навчилася специфічній термінології та контексту.

### 3. Великі Моделі та Обмеження Пам'яті

**Base BERT:** 110M parameters, максимальна довжина 512 токенів.

**Проблеми:**
- Квадратична залежність пам'яті від довжини послідовності (див. аналіз вище)
- Для production потрібні оптимізації:
  - **DistilBERT:** 66M parameters, ~2× швидше, зберігає ~97% точності
  - **TinyBERT:** Ще менша модель для edge devices
  - **MobileBERT:** Оптимізована для мобільних пристроїв

**Практичні рекомендації:**
- Для коротких логів (< 512 токенів): BERT-base або DistilBERT
- Для довгих логів (> 512 токенів): Longformer або BigBird
- Для production з обмеженими ресурсами: DistilBERT або TinyBERT

## Ключові Висновки

1. **Self-Attention:** Дозволяє моделі "дивитися" на всі слова одночасно та зважувати їх важливість.

2. **Контекстні Embeddings:** Кожне слово має різний вектор залежно від контексту, що вирішує проблему полісемії.

3. **`[CLS]` токен:** Кодує представлення всього речення для класифікації.

4. **Bidirectional:** BERT бачить контекст з обох сторін, на відміну від односторонніх моделей.

5. **Краще за Naive Bayes:** Явно моделює залежності між словами, не припускає незалежність.

6. **Зв'язок з Теоремою Байєса:** Мінімізація крос-ентропії в BERT еквівалентна максимізації лог-ймовірності (Maximum Likelihood), що є ймовірнісним підходом, подібним до теореми Байєса.

У наступному розділі ми подивимося, як генерувати синтетичні дані для навчання та тестування цих моделей.

## Рекомендована Література

### Оригінальні Роботи про Трансформери

1. **Vaswani, A., et al.** (2017). "Attention is All You Need"
   - NIPS. Оригінальна робота про архітектуру Transformer.

2. **Devlin, J., et al.** (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
   - NAACL. Оригінальна робота про BERT.

### Self-Attention та Механізми

3. **Bahdanau, D., Cho, K., & Bengio, Y.** (2014). "Neural Machine Translation by Jointly Learning to Align and Translate"
   - arXiv:1409.0473. Перша робота про attention в NLP.

4. **Lin, Z., et al.** (2017). "A Structured Self-Attentive Sentence Embedding"
   - arXiv:1703.03130. Self-attention для представлення речень.

### BERT та Fine-tuning

5. **Howard, J., & Ruder, S.** (2018). "Universal Language Model Fine-tuning for Text Classification"
   - ACL. ULMFiT — попередник BERT для fine-tuning.

6. **Sun, C., et al.** (2019). "How to Fine-Tune BERT for Text Classification?"
   - Chinese Computational Linguistics. Практичний гайд з fine-tuning BERT.

### Оптимізації та Альтернативи

7. **Sanh, V., et al.** (2019). "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"
   - NeurIPS. Оптимізація BERT через distillation.

8. **Beltagy, I., Peters, M. E., & Cohan, A.** (2020). "Longformer: The Long-Document Transformer"
   - arXiv:2004.05150. BERT для довгих документів.

### Практична Реалізація

9. **Wolf, T., et al.** (2020). "Transformers: State-of-the-Art Natural Language Processing"
   - EMNLP. HuggingFace Transformers library.

10. **HuggingFace Documentation**
    - URL: https://huggingface.co/docs/transformers
    - Практичний гайд з використанням BERT та інших моделей.

### Теоретичне Розуміння

11. **Rogers, A., Kovaleva, O., & Rumshisky, A.** (2020). "A Primer in BERTology: What we know about how BERT works"
    - TACL. Огляд того, що ми знаємо про BERT.

12. **Clark, K., et al.** (2019). "What Does BERT Look At? An Analysis of BERT's Attention"
    - ACL. Аналіз attention weights в BERT.

### Інформаційна Ентропія та Крос-Ентропія

13. **Shannon, C. E.** (1948). "A Mathematical Theory of Communication"
    - Bell System Technical Journal, 27(3), 379-423. Оригінальна робота про інформаційну ентропію.

14. **Cover, T. M., & Thomas, J. A.** (2006). "Elements of Information Theory"
    - 2nd Edition. Wiley. Класичний підручник про теорію інформації, включаючи ентропію та крос-ентропію.

15. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). "Deep Learning"
    - MIT Press. Розділ 5.5: "Maximum Likelihood Estimation" та Розділ 6.2: "Cost Functions". Зв'язок між максимізацією ймовірності та мінімізацією крос-ентропії.

16. **Bishop, C. M.** (2006). "Pattern Recognition and Machine Learning"
    - Springer. Розділ 1.6: "Information Theory" та Розділ 4.3: "Probabilistic Discriminative Models". Математичне обґрунтування крос-ентропії.

17. **Murphy, K. P.** (2022). "Probabilistic Machine Learning: An Introduction"
    - MIT Press. Розділ 5.1: "Maximum Likelihood Estimation" та Розділ 5.4: "Cross-Entropy Loss". Детальне пояснення зв'язку між теоремою Байєса та крос-ентропією.

---

**Примітка для студентів:** Почніть з Vaswani et al. для розуміння Transformer, потім перейдіть до Devlin et al. для BERT. Для розуміння математичного зв'язку між теоремою Байєса та крос-ентропією прочитайте Goodfellow et al. (розділ 5.5) та Bishop (розділ 1.6 та 4.3). Для практики використовуйте HuggingFace Transformers та приклади з Sun et al. Для глибшого розуміння прочитайте Rogers et al. про BERTology.
