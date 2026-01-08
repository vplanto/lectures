---
title: "Механізм Attention: Фокус на важливому без рекурсії"
layout: default
nav_order: 4.5
parent: "Блок 2: Архітектура Пам'яті (Deep Learning)"
---

# Механізм Attention: Фокус на важливому без рекурсії

## Парадокс: Чому LSTM все ще обмежена

Уявіть ситуацію: ваш LSTM навчений на логах за тиждень (10,080 хвилин). Для передбачення падіння сервера критична подія сталася 3 дні тому (4,320 хвилин тому). LSTM має пройти через всі 4,320 кроків, щоб "донести" цю інформацію до поточного моменту. Навіть з механізмом gates, інформація може "розмитися" або "забутися" під час цього довгого шляху.

**Проблема:** LSTM все ще вимагає **послідовного проходження** через весь ланцюг часу, навіть якщо важлива інформація знаходиться далеко в минулому.

**Рішення:** Механізм **Attention** дозволяє моделі **безпосередньо звертатися** до будь-якого моменту в історії, не проходячи через всі проміжні кроки. Це як мати "прямий доступ" до архіву подій замість перегляду всіх записів по черзі.

## Математичний фундамент

### Обмеження LSTM

**Проблема 1: Послідовна обробка**
- LSTM обробляє послідовність крок за кроком: $h_t = \text{LSTM}(x_t, h_{t-1})$
- Для доступу до інформації з кроку $t-k$ потрібно пройти через $k$ кроків
- Інформація може "розмитися" через багато gates

**Проблема 2: Фіксована ємність пам'яті**
- Hidden state $h_t$ має фіксований розмір (наприклад, 128 розмірностей)
- Вся інформація з історії має "поміститися" в цей вектор
- При довгих послідовностях важлива інформація може "витіснятися"

**Проблема 3: Однакова увага до всіх кроків**
- LSTM не може "фокусуватися" на конкретних важливих моментах
- Всі кроки обробляються однаково, незалежно від їх релевантності

### Механізм Attention: Основна ідея

**Ключова концепція:** Замість того, щоб стискати всю історію в один вектор $h_t$, зберігаємо **всі попередні hidden states** $H = [h_1, h_2, \ldots, h_{t-1}]$ і дозволяємо моделі **вибирати**, які з них використовувати для поточного передбачення.

**Формально:**

Для поточного кроку $t$ та всіх попередніх hidden states $H = \{h_1, h_2, \ldots, h_{t-1}\}$:

1. **Обчислюємо ваги уваги (attention weights):**
   $$\alpha_{t,i} = \text{softmax}(\text{score}(h_t, h_i))$$

2. **Обчислюємо контекстний вектор (context vector):**
   $$c_t = \sum_{i=1}^{t-1} \alpha_{t,i} \cdot h_i$$

3. **Використовуємо контекст для передбачення:**
   $$\hat{y}_t = f(h_t, c_t)$$

### Scaled Dot-Product Attention

Найпоширеніший варіант attention, запропонований у статті "Attention is All You Need" (Vaswani et al., 2017):

**Крок 1: Query, Key, Value**

Для кожного кроку $i$:
- **Query** $q_i = W_q h_i$ — "питання": що ми шукаємо?
- **Key** $k_i = W_k h_i$ — "ключ": що цей крок може запропонувати?
- **Value** $v_i = W_v h_i$ — "значення": що цей крок містить?

**Крок 2: Обчислення attention scores**

Для поточного кроку $t$ та всіх попередніх $i \in [1, t-1]$:

$$\text{score}(q_t, k_i) = \frac{q_t^T k_i}{\sqrt{d_k}}$$

Де $d_k$ — розмірність ключів (для стабілізації градієнтів).

**Крок 3: Softmax нормалізація**

$$\alpha_{t,i} = \frac{\exp(\text{score}(q_t, k_i))}{\sum_{j=1}^{t-1} \exp(\text{score}(q_t, k_j))}$$

**Крок 4: Зважена сума values**

$$c_t = \sum_{i=1}^{t-1} \alpha_{t,i} \cdot v_i$$

**Повна формула Scaled Dot-Product Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Де:
- $Q$ — матриця queries (розмір $n \times d_k$)
- $K$ — матриця keys (розмір $m \times d_k$)
- $V$ — матриця values (розмір $m \times d_v$)

### Multi-Head Attention

Для захоплення різних типів залежностей використовується **Multi-Head Attention**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Де кожен "голова" (head) обчислює attention незалежно:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Інтерпретація:**
- Кожна "голова" може фокусуватися на різних аспектах: короткострокові тренди, сезонність, аномалії тощо
- Типово використовується $h = 8$ голів

## Інженерна інтерпретація

### Аналогія: Пошук в архіві vs Читання всіх документів

**LSTM (без Attention):**
- Як читати всі документи по черзі від початку до кінця
- Якщо важливий документ на сторінці 1000, потрібно прочитати всі попередні 999
- Інформація може "забуватися" під час читання

**LSTM з Attention:**
- Як мати індекс з ключовими словами та можливість "стрибати" до потрібного документа
- Можна одразу звернутися до сторінки 1000, якщо вона релевантна
- Можна використовувати кілька документів одночасно, зважуючи їх важливість

### Практичний приклад: Виявлення аномалії в логах

**Сценарій:** Аналіз логів сервера за тиждень для виявлення причини падіння.

**Події:**
- **День 1 (крок 1):** Початок деградації бази даних (критично!)
- **День 2-6 (кроки 2-6):** Нормальна робота
- **День 7 (крок 7):** Падіння сервера

**LSTM без Attention:**
- Прихований стан $h_7$ містить "стиснуту" інформацію з усіх 7 днів
- Інформація про день 1 може бути "розмитою" або "забутою"
- Модель може не змогти встановити зв'язок між днем 1 та днем 7

**LSTM з Attention:**
- Модель обчислює attention weights $\alpha_{7,1}, \alpha_{7,2}, \ldots, \alpha_{7,6}$
- Якщо $\alpha_{7,1}$ великий (наприклад, 0.6), модель "фокусується" на події з дня 1
- Контекстний вектор $c_7$ містить переважно інформацію з дня 1
- Модель успішно встановлює зв'язок між початком деградації та падінням

### Переваги Attention для SRE

1. **Прямий доступ до історії:** Не потрібно проходити через всі проміжні кроки
2. **Інтерпретативність:** Attention weights показують, які моменти важливі
3. **Паралелізація:** Обчислення attention можна паралелізувати (на відміну від послідовної обробки LSTM)
4. **Гнучкість:** Модель може фокусуватися на різних частинах історії для різних передбачень

## Реалізація на Python

### Реалізація Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention механізм.
    """
    def __init__(self, d_model: int, d_k: Optional[int] = None):
        """
        Parameters:
        -----------
        d_model : int
            Розмірність моделі (розмірність hidden states)
        d_k : int, optional
            Розмірність ключів/запитів (за замовчуванням d_model)
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k if d_k is not None else d_model
        self.d_model = d_model
        
        # Лінійні проекції для Query, Key, Value
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, d_model)
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass attention механізму.
        
        Parameters:
        -----------
        queries : torch.Tensor
            Матриця запитів (batch_size, seq_len_q, d_k)
        keys : torch.Tensor
            Матриця ключів (batch_size, seq_len_k, d_k)
        values : torch.Tensor
            Матриця значень (batch_size, seq_len_v, d_model)
        mask : torch.Tensor, optional
            Маска для приховування певних позицій (batch_size, seq_len_q, seq_len_k)
        
        Returns:
        --------
        output : torch.Tensor
            Вихід attention (batch_size, seq_len_q, d_model)
        attention_weights : torch.Tensor
            Ваги уваги (batch_size, seq_len_q, seq_len_k)
        """
        batch_size = queries.size(0)
        
        # Проекції
        Q = self.W_q(queries)  # (batch_size, seq_len_q, d_k)
        K = self.W_k(keys)    # (batch_size, seq_len_k, d_k)
        V = self.W_v(values)  # (batch_size, seq_len_v, d_model)
        
        # Обчислення scores: QK^T / sqrt(d_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.d_k)
        # scores shape: (batch_size, seq_len_q, seq_len_k)
        
        # Застосування маски (якщо є)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax для отримання attention weights
        attention_weights = F.softmax(scores, dim=-1)
        # attention_weights shape: (batch_size, seq_len_q, seq_len_k)
        
        # Зважена сума values
        output = torch.bmm(attention_weights, V)
        # output shape: (batch_size, seq_len_q, d_model)
        
        return output, attention_weights
```

### LSTM з Attention для прогнозування

```python
class LSTMAttentionForecaster(nn.Module):
    """
    LSTM модель з механізмом Attention для прогнозування часових рядів.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 1, 
                 attention_dim: Optional[int] = None):
        """
        Parameters:
        -----------
        input_size : int
            Розмірність входу
        hidden_size : int
            Розмірність прихованого стану LSTM
        num_layers : int
            Кількість шарів LSTM
        output_size : int
            Розмірність виходу
        attention_dim : int, optional
            Розмірність для attention (за замовчуванням hidden_size)
        """
        super(LSTMAttentionForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_dim = attention_dim if attention_dim is not None else hidden_size
        
        # LSTM шар
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        
        # Attention механізм
        self.attention = ScaledDotProductAttention(
            d_model=hidden_size, 
            d_k=self.attention_dim
        )
        
        # Вихідний шар
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Вхідна послідовність (batch_size, seq_len, input_size)
        return_attention : bool
            Чи повертати attention weights для візуалізації
        
        Returns:
        --------
        output : torch.Tensor
            Передбачення (batch_size, output_size)
        attention_weights : torch.Tensor, optional
            Ваги уваги (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # Використовуємо останній hidden state як query
        # Для self-attention використовуємо всі hidden states
        query = lstm_out[:, -1:, :]  # (batch_size, 1, hidden_size)
        keys = lstm_out  # (batch_size, seq_len, hidden_size)
        values = lstm_out  # (batch_size, seq_len, hidden_size)
        
        # Attention
        context, attention_weights = self.attention(query, keys, values)
        # context shape: (batch_size, 1, hidden_size)
        # attention_weights shape: (batch_size, 1, seq_len)
        
        # Вихідний шар
        output = self.fc(context.squeeze(1))  # (batch_size, output_size)
        
        if return_attention:
            return output, attention_weights.squeeze(1)  # (batch_size, seq_len)
        return output
```

### Порівняння LSTM з та без Attention

```python
def generate_anomaly_dataset(n_samples: int = 1000, seq_length: int = 100, 
                            anomaly_position: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерує датасет з аномалією на певній позиції.
    
    Задача: Передбачити значення на основі послідовності, де важлива подія
    сталася далеко в минулому.
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Нормальний часовий ряд
        sequence = np.random.normal(0, 0.1, seq_length)
        
        # Додаємо аномалію на позиції anomaly_position
        sequence[anomaly_position] = 2.0  # Великий сплеск
        
        # Цільове значення залежить від аномалії
        # Якщо аномалія була, то очікуємо високе значення
        target = 1.0 if sequence[anomaly_position] > 1.5 else 0.0
        
        X.append(sequence.reshape(-1, 1))
        y.append(target)
    
    return np.array(X), np.array(y)

# Генерація датасету
X_train, y_train = generate_anomaly_dataset(n_samples=800, seq_length=50, anomaly_position=10)
X_test, y_test = generate_anomaly_dataset(n_samples=200, seq_length=50, anomaly_position=10)

# Конвертація в тензори
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# Модель 1: LSTM без Attention
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Модель 2: LSTM з Attention
model_lstm = SimpleLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
model_lstm_attention = LSTMAttentionForecaster(
    input_size=1, hidden_size=64, num_layers=2, output_size=1
)

# Навчання обох моделей
def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    """Навчає модель та повертає історію втрат."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Навчання
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Оцінка
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train)
            test_pred = model(X_test)
            train_loss = criterion(train_pred, y_train).item()
            test_loss = criterion(test_pred, y_test).item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    return train_losses, test_losses

print("Навчання LSTM без Attention...")
train_losses_lstm, test_losses_lstm = train_model(
    model_lstm, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
)

print("\nНавчання LSTM з Attention...")
train_losses_attn, test_losses_attn = train_model(
    model_lstm_attention, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
)

# Порівняння
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses_lstm, label='LSTM (train)', linewidth=2)
plt.plot(test_losses_lstm, label='LSTM (test)', linewidth=2, linestyle='--')
plt.plot(train_losses_attn, label='LSTM+Attention (train)', linewidth=2)
plt.plot(test_losses_attn, label='LSTM+Attention (test)', linewidth=2, linestyle='--')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Порівняння навчання', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Фінальні помилки
model_lstm.eval()
model_lstm_attention.eval()
with torch.no_grad():
    pred_lstm = model_lstm(X_test_tensor)
    pred_attn, attention_weights = model_lstm_attention(X_test_tensor, return_attention=True)
    
    mse_lstm = nn.MSELoss()(pred_lstm, y_test_tensor).item()
    mse_attn = nn.MSELoss()(pred_attn, y_test_tensor).item()

plt.subplot(1, 2, 2)
models = ['LSTM', 'LSTM+Attention']
mses = [mse_lstm, mse_attn]
colors = ['blue', 'green']
bars = plt.bar(models, mses, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Test MSE', fontsize=12)
plt.title('Фінальна помилка на тесті', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')

# Додаємо значення на стовпцях
for bar, mse in zip(bars, mses):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{mse:.4f}',
             ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('lstm_vs_attention.png', dpi=300)
plt.show()

print(f"\n{'='*60}")
print("Порівняння моделей:")
print(f"{'='*60}")
print(f"LSTM (без Attention): Test MSE = {mse_lstm:.6f}")
print(f"LSTM (з Attention):  Test MSE = {mse_attn:.6f}")
print(f"Покращення: {mse_lstm / mse_attn:.2f}x")
```

### Візуалізація Attention Weights

```python
def visualize_attention(sequence: np.ndarray, attention_weights: np.ndarray, 
                       anomaly_position: int, title: str = "Attention Weights"):
    """
    Візуалізує послідовність та attention weights.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Графік 1: Послідовність
    axes[0].plot(sequence, 'b-', linewidth=2, label='Послідовність', alpha=0.7)
    axes[0].axvline(x=anomaly_position, color='r', linestyle='--', 
                   linewidth=2, label=f'Аномалія (позиція {anomaly_position})')
    axes[0].scatter([anomaly_position], [sequence[anomaly_position]], 
                   color='red', s=100, zorder=5, marker='*')
    axes[0].set_xlabel('Позиція в послідовності', fontsize=12)
    axes[0].set_ylabel('Значення', fontsize=12)
    axes[0].set_title('Вхідна послідовність', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Графік 2: Attention weights
    axes[1].bar(range(len(attention_weights)), attention_weights, 
               alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=anomaly_position, color='r', linestyle='--', 
                   linewidth=2, label=f'Аномалія (позиція {anomaly_position})')
    axes[1].set_xlabel('Позиція в послідовності', fontsize=12)
    axes[1].set_ylabel('Attention Weight', fontsize=12)
    axes[1].set_title(title, fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Знаходимо позицію з максимальною увагою
    max_attention_idx = np.argmax(attention_weights)
    max_attention_val = attention_weights[max_attention_idx]
    axes[1].text(0.02, 0.98, 
               f'Максимальна увага: позиція {max_attention_idx} (weight = {max_attention_val:.4f})',
               transform=axes[1].transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
               fontsize=11)
    
    plt.tight_layout()
    return fig

# Візуалізація для кількох прикладів
model_lstm_attention.eval()
with torch.no_grad():
    # Беремо перші 5 прикладів
    for i in range(min(5, len(X_test))):
        sample = X_test[i:i+1]
        sample_tensor = torch.FloatTensor(sample)
        
        _, attention_weights = model_lstm_attention(sample_tensor, return_attention=True)
        attention_weights_np = attention_weights[0].numpy()
        
        fig = visualize_attention(
            X_test[i, :, 0], 
            attention_weights_np,
            anomaly_position=10,
            title=f'Attention Weights для прикладу {i+1}'
        )
        plt.savefig(f'attention_example_{i+1}.png', dpi=300)
        plt.show()
        
        print(f"\nПриклад {i+1}:")
        print(f"  Позиція аномалії: {10}")
        print(f"  Позиція з максимальною увагою: {np.argmax(attention_weights_np)}")
        print(f"  Attention weight на позиції аномалії: {attention_weights_np[10]:.4f}")
        print(f"  Максимальний attention weight: {np.max(attention_weights_np):.4f}")
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention механізм.
    """
    def __init__(self, d_model: int, num_heads: int = 8):
        """
        Parameters:
        -----------
        d_model : int
            Розмірність моделі
        num_heads : int
            Кількість "голів" attention
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model повинен ділитися на num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Проекції для всіх голів
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass Multi-Head Attention.
        
        Returns:
        --------
        output : torch.Tensor
            Вихід (batch_size, seq_len_q, d_model)
        attention_weights : torch.Tensor
            Ваги уваги (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = queries.size(0)
        seq_len_q = queries.size(1)
        seq_len_k = keys.size(1)
        
        # Проекції та розділення на голови
        Q = self.W_q(queries).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(keys).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(values).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, d_k)
        
        # Scaled Dot-Product Attention для кожної голови
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Зважена сума
        context = torch.matmul(attention_weights, V)
        # Shape: (batch_size, num_heads, seq_len_q, d_k)
        
        # Об'єднання голів
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        # Вихідна проекція
        output = self.W_o(context)
        
        return output, attention_weights

# Приклад використання Multi-Head Attention
class LSTMMultiHeadAttention(nn.Module):
    """LSTM з Multi-Head Attention."""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, 
                 output_size=1, num_heads=8):
        super(LSTMMultiHeadAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.multi_head_attention = MultiHeadAttention(hidden_size, num_heads)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        query = lstm_out[:, -1:, :]
        context, attention_weights = self.multi_head_attention(query, lstm_out, lstm_out)
        output = self.fc(context.squeeze(1))
        
        if return_attention:
            return output, attention_weights
        return output
```

## Висновки та наступні кроки

**Ключові висновки:**

1. **Attention вирішує обмеження LSTM:** Дозволяє моделі безпосередньо звертатися до будь-якого моменту в історії
2. **Інтерпретативність:** Attention weights показують, які моменти важливі для передбачення
3. **Паралелізація:** Обчислення attention можна паралелізувати (на відміну від послідовної обробки LSTM)
4. **Гнучкість:** Multi-Head Attention дозволяє фокусуватися на різних аспектах одночасно

**Порівняння архітектур:**

| Архітектура | Обробка | Доступ до історії | Паралелізація | Інтерпретативність |
|------------|---------|-------------------|---------------|-------------------|
| RNN | Послідовна | Через всі кроки | Немає | Низька |
| LSTM | Послідовна | Через gates | Немає | Середня |
| LSTM + Attention | Гібридна | Прямий доступ | Часткова | Висока |
| Transformer | Паралельна | Прямий доступ | Повна | Висока |

**Для SRE практики:**
- Використовуйте Attention для задач, де важливі події можуть бути далеко в минулому
- Attention weights можна використовувати для діагностики: які моменти модель вважає важливими?
- Multi-Head Attention дозволяє захопити різні типи залежностей (тренди, сезонність, аномалії)

**Наступні кроки:**
- Transformer архітектура (повністю заснована на Attention, без рекурсії)
- Self-Attention для часових рядів
- Temporal Convolutional Networks (TCN) як альтернатива RNN/LSTM

---

## Додаткові матеріали

### Рекомендована література

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural machine translation by jointly learning to align and translate." *arXiv preprint arXiv:1409.0473*.
2. Vaswani, A., et al. (2017). "Attention is all you need." *Advances in neural information processing systems*, 30.
3. Luong, M. T., Pham, H., & Manning, C. D. (2015). "Effective approaches to attention-based neural machine translation." *arXiv preprint arXiv:1508.04025*.

### Вправи для самостійної роботи

1. **Завдання 1:** Реалізуйте **Self-Attention** для часових рядів, де кожен крок уважає всі інші кроки в послідовності. Порівняйте з LSTM+Attention на задачі прогнозування.

2. **Завдання 2:** Дослідіть вплив **кількості голів** у Multi-Head Attention на якість моделі. Покажіть, як різні голови фокусуються на різних аспектах (тренди, сезонність, аномалії).

3. **Завдання 3:** Створіть **візуалізацію attention heatmap** для реальних метрик (CPU, memory, latency). Покажіть, які історичні моменти модель вважає важливими для передбачення аномалій.


