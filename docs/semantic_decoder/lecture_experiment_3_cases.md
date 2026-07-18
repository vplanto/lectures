# Лекція 4: Експериментальне підтвердження (Тестування 3-х кейсів)

## Вступ: Архітектура експерименту
Для перевірки ефективності семантичного розрізнення було розроблено програмний комплекс, який моделює передачу даних (10 можливих станів) через різні типи каналів. Щоб довести життєздатність підходу Андрія, ми проганяємо алгоритм через три фундаментально різні сценарії.

---

## Кейс 1: Нормальна ситуація (Бейзлайн)
**Умови:** Канал зв'язку працює ідеально, матриця перехідних ймовірностей $P(Y|X)$ має повний ранг (немає виродженості).
**Результат:** Як класичний статистичний декодер, так і наша нейромережева модель з Embeddings та GRU легко вивчають прямі відповідності. 
**Висновок:** Новий метод працює стабільно і дає таку ж високу точність, як і звичайні методи. Він не ламає те, що вже добре працює.

---

## Кейс 2: Випадкові дані (Контрольна вибірка / Хаос)
**Умови:** На вхід подається послідовність абсолютно незалежних подій (Memoryless source). Канал може бути зашумленим, але найголовніше — між сигналами немає жодної часової зв'язності (контексту).
**Результат:** Оскільки прихованих залежностей (траєкторії) не існує, рекурентній мережі (GRU) немає за що зачепитися. Семантичний підхід теоретично не може надати жодних переваг.
**Висновок:** Хаос опанувати неможливо. Залишкова ентропія залишається максимальною, і новий метод чесно показує відсутність результату, що підтверджує його фокус саме на пошуку сенсу, а не на випадкових вгадуваннях.

---

## Кейс 3: Цільовий вироджений кейс (Де класика "ламається")
**Умови:** 1. Створено складну вироджену матрицю (наприклад, розміром 10x10, але з математичним рангом 5 або 8). Різні вхідні стани відображаються в ідентичні вихідні розподіли.
2. Дані є структурованими — вони містять латентні часові залежності (контекст).

**Результат (Магія в дії):**
* **Провал класики:** Через нульову кодову відстань звичайний ML-декодер повністю сліпне. Його точність падає.
* **Тріумф семантики:** Завдяки аналізу послідовностей (вікно $L=20$) та використанню 16-вимірного латентного простору, модель успішно "розклеює" вироджені стани. 
* **Статистичне підтвердження:** За результатами 5 прогонів експерименту точність семантичного методу є статистично значуще вищою (за T-тестом $p < 0.05$) порівняно з класикою. Залишкова ентропія різко падає нижче базового рівня $H_{baseline}$.

---

## Візуалізація результатів
Якщо подивитися на згенеровані графіки розсіювання (Scatter Plots) для фінального геометричного розподілу станів у 2D, ми побачимо наочне підтвердження:
Стани, які для класичного алгоритму виглядали як одна математична точка (через однакові ймовірності), у семантичному просторі чітко розведені по різних координатах (кластерах). Алгоритм орієнтується на відстань між цими точками (Cosine Similarity), ігноруючи початкову невизначеність каналу.

---

## Покроковий план реалізації для Google Colab

> Оригінальний код (`tik.py`) має три системні проблеми, які заважають коректно відтворити всі три кейси:
> 1. **Нестабільна випадковість** — матриця генерується рандомно, результати непередбачувані між запусками.
> 2. **Відсутність зниження розмірності** — scatter plot малює лише перші 2 виміри з 8, кластери можуть бути невидимі. Потрібен PCA або t-SNE.
> 3. **Відсутність Кейсу 1 і Кейсу 2** — у коді є лише один вироджений сценарій.
>
> Нижче описано план, що усуває всі три проблеми.

---

### Крок 1. Встановлення та імпорти

```python
# У Colab всі бібліотеки вже є, але для чистоти:
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
import math

# Фіксуємо всі генератори випадкових чисел для відтворюваності
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```

**Чому важливо:** `np.random.seed(SEED)` + `torch.manual_seed(SEED)` вирішують **Проблему 1** — тепер матриця і навчання дають однаковий результат при кожному запуску.

---

### Крок 2. Генерація трьох детерміністичних матриць

```python
N_STATES = 10

def make_full_rank_matrix(n=10):
    """Кейс 1: Нормальна матриця з повним рангом."""
    rng = np.random.default_rng(SEED)
    m = rng.dirichlet(np.ones(n) * 5, size=n)  # Сильна діагональ
    return m

def make_degenerate_matrix(n=10, rank=5):
    """Кейс 3: Вироджена матриця з фіксованим рангом."""
    rng = np.random.default_rng(SEED)
    A = rng.random((n, rank))
    B = rng.random((rank, n))
    m = A @ B
    row_sums = m.sum(axis=1, keepdims=True)
    return m / row_sums

P_normal     = make_full_rank_matrix()   # Кейс 1
P_degenerate = make_degenerate_matrix()  # Кейс 3

print(f"Ранг нормальної матриці:   {np.linalg.matrix_rank(P_normal)}")
print(f"Ранг виродженої матриці:   {np.linalg.matrix_rank(P_degenerate)}")
```

**Ключова різниця від `tik.py`:** Замість `np.random.rand()` використовується `np.random.default_rng(SEED)` — це дає стабільні матриці при кожному запуску.

---

### Крок 3. Генерація трьох датасетів (по одному на кейс)

```python
def generate_structured(matrix, n_states, size=80000):
    """Кейс 1 і 3: Структурована послідовність (X залежить від попереднього X)."""
    rng = np.random.default_rng(SEED)
    X, Y = [], []
    curr_x = rng.integers(n_states // 2)
    for _ in range(size):
        # Пара: X < 5 → наступний X = curr_x + 5
        next_x = (curr_x + n_states // 2) % n_states if curr_x < n_states // 2 else rng.integers(n_states // 2)
        y = rng.choice(n_states, p=matrix[next_x])
        X.append(next_x); Y.append(float(y))
        curr_x = next_x
    return np.array(X), np.array(Y).reshape(-1, 1)

def generate_random(matrix, n_states, size=20000):
    """Кейс 2: Хаотична послідовність (Memoryless source)."""
    rng = np.random.default_rng(SEED + 1)
    X = rng.integers(n_states, size=size)
    Y = np.array([rng.choice(n_states, p=matrix[x]) for x in X], dtype=float).reshape(-1, 1)
    return X, Y

# Генеруємо
X_case1, Y_case1 = generate_structured(P_normal,     N_STATES, size=80000)
X_case2, Y_case2 = generate_random    (P_degenerate, N_STATES, size=20000)
X_case3, Y_case3 = generate_structured(P_degenerate, N_STATES, size=80000)

print("Датасети готові")
```

**Виправлення Проблеми 3:** Тепер є три окремих датасети з чітко різною структурою — нормальна матриця зі структурою, хаотичні дані, вироджена матриця зі структурою.

---

### Крок 4. Підготовка послідовностей та DataLoader

```python
SEQ_LEN = 20

def to_sequences(Y, X, seq_len=SEQ_LEN):
    Y_seq = [Y[i:i+seq_len].flatten() for i in range(len(Y) - seq_len)]
    X_tgt = [X[i + seq_len - 1]       for i in range(len(X) - seq_len)]
    return torch.LongTensor(np.array(Y_seq)), torch.LongTensor(np.array(X_tgt))

def make_loaders(X, Y, split=0.8, batch_size=2048):
    seqs, tgts = to_sequences(Y, X)
    n = int(len(seqs) * split)
    train_ds = TensorDataset(seqs[:n], tgts[:n])
    test_ds  = TensorDataset(seqs[n:], tgts[n:])
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
           DataLoader(test_ds,  batch_size=batch_size)

loader_c1_tr, loader_c1_te = make_loaders(X_case1, Y_case1)
loader_c2_tr, loader_c2_te = make_loaders(X_case2, Y_case2)
loader_c3_tr, loader_c3_te = make_loaders(X_case3, Y_case3)
```

---

### Крок 5. Модель SemanticDecoder

```python
class SemanticDecoder(nn.Module):
    def __init__(self, n_states, embed_dim=16, hidden=64):
        super().__init__()
        self.embeddings = nn.Embedding(n_states, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden, num_layers=2,
                          batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, n_states)

    def forward(self, x):
        x = self.embeddings(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])   # беремо останній крок
```

**Відмінність від `tik.py`:** `embed_dim=16` замість 8 (відповідає опису в лекції "16-вимірний латентний простір").

---

### Крок 6. Функція тренування одного кейсу

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader, n_epochs=80, seed=SEED):
    torch.manual_seed(seed)
    model = SemanticDecoder(N_STATES).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(n_epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()
    return model

def evaluate(model, test_loader):
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            logits = model(bx.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.extend(probs.argmax(axis=1))
            all_true.extend(by.numpy())
    all_probs = np.vstack(all_probs)
    acc = accuracy_score(all_true, all_preds)
    ent = -np.mean(np.sum(all_probs * np.log2(all_probs + 1e-12), axis=1))
    return acc, ent, model
```

---

### Крок 7. Запуск трьох кейсів та порівняльна таблиця

```python
H_max = math.log2(N_STATES)

results = {}
for name, tr_loader, te_loader in [
    ("Кейс 1 (Норма)",       loader_c1_tr, loader_c1_te),
    ("Кейс 2 (Хаос)",        loader_c2_tr, loader_c2_te),
    ("Кейс 3 (Вироджений)",  loader_c3_tr, loader_c3_te),
]:
    print(f"\n=== {name} ===")
    model = train_model(tr_loader)
    acc, ent, model = evaluate(model, te_loader)
    results[name] = {"model": model, "accuracy": acc, "entropy": ent}
    print(f"Accuracy: {acc:.4f}  |  Залишкова ентропія: {ent:.4f} / {H_max:.2f} біт")
```

**Очікувані результати:**

| Кейс | Accuracy | Ентропія |
|---|---|---|
| Кейс 1 (Норма) | ~0.90+ | низька |
| Кейс 2 (Хаос) | ~0.10 (=1/10) | ~3.32 біт (максимум) |
| Кейс 3 (Вироджений) | >0.50 (значно вище класики) | нижче max |

---

### Крок 8. Візуалізація кластерів через PCA (виправлення Проблеми 2)

```python
# Виправлення: замість перших 2 вимірів — PCA
def plot_clusters(model, title):
    weights = model.embeddings.weight.detach().cpu().numpy()  # (10, 16)
    coords  = PCA(n_components=2, random_state=SEED).fit_transform(weights)

    plt.figure(figsize=(8, 6))
    for i in range(N_STATES):
        color = 'crimson' if i < 5 else 'royalblue'
        plt.scatter(coords[i, 0], coords[i, 1], c=color, s=200, zorder=3)
        plt.text(coords[i, 0] + 0.02, coords[i, 1] + 0.02, f'X{i}', fontsize=11)
    plt.title(title)
    plt.xlabel("PCA-1"); plt.ylabel("PCA-2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_clusters(results["Кейс 1 (Норма)"]["model"],      "Кейс 1: Кластери (Нормальний канал)")
plot_clusters(results["Кейс 2 (Хаос)"]["model"],       "Кейс 2: Кластери (Хаотичні дані)")
plot_clusters(results["Кейс 3 (Вироджений)"]["model"], "Кейс 3: Кластери (Вироджений канал)")
```

**Чому PCA, а не прямі ваги:** Ембедінг-простір 16-вимірний. Перші 2 ваги (`weights[:,0]` та `weights[:,1]`) можуть не містити найважливішу варіацію. PCA проектує дані у напрямку **максимальної дисперсії**, тому кластери будуть видимі навіть якщо семантика "захована" у пізніших вимірах.

