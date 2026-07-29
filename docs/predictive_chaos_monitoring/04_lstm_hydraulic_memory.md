---
title: "04 Lstm Hydraulic Memory"
type: lecture
module: Module 4
prerequisites: module 3
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# LSTM як "Гідравлічна система" управління інформацією

## Парадокс: Як "заблукати" інформацію, щоб її не забути

Уявіть гідравлічну систему з трьома вентилями:
- **Forget Gate (Вентиль забуття):** Контролює, скільки старої води (інформації) випустити
- **Input Gate (Вхідний вентиль):** Контролює, скільки нової води впустити
- **Output Gate (Вихідний вентиль):** Контролює, скільки води випустити назовні

Це не метафора — це точний опис архітектури **LSTM (Long Short-Term Memory)**, розробленої в 1997 році для вирішення проблеми зникаючого градієнта. LSTM не просто "запам'ятовує" інформацію — вона **активно керує** потоком інформації через час, вирішуючи, що забути, що запам'ятати, і що передати далі.

## Математичний фундамент

### Архітектура LSTM комірки

**LSTM комірка** складається з трьох основних компонентів:

1. **Cell State** $C_t$ — "конвеєрна стрічка" інформації
2. **Hidden State** $h_t$ — вихід комірки
3. **Gates** — механізми контролю потоку

### Детальний розбір формул

**Крок 1: Forget Gate (Вентиль забуття)**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

Де:
- $\sigma$ — сигмоїдна функція: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- $W_f$ — матриця ваг forget gate
- $b_f$ — зміщення
- $[h_{t-1}, x_t]$ — конкатенація попереднього прихованого стану та поточного входу

**Інтерпретація:** $f_t \in [0, 1]$ визначає, скільки інформації з $C_{t-1}$ зберегти:
- $f_t \approx 0$: "Забути все"
- $f_t \approx 1$: "Зберегти все"

**Крок 2: Input Gate (Вхідний вентиль)**

Визначає, яку нову інформацію додати до cell state:

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

Де:
- $i_t$ — скільки нової інформації додати
- $\tilde{C}_t$ — кандидат на нові значення cell state

**Крок 3: Оновлення Cell State**

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

Де $\odot$ — поелементне множення (Hadamard product).

**Інтерпретація:**
- $f_t \odot C_{t-1}$: Зберігаємо частину старої інформації
- $i_t \odot \tilde{C}_t$: Додаємо нову інформацію

**Крок 4: Output Gate (Вихідний вентиль)**

Визначає, яку частину cell state використати для виходу:

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

**Повна система рівнянь LSTM:**

$$\begin{aligned}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}$$

### Чому це вирішує проблему зникаючого градієнта

**Ключова ідея:** Cell state $C_t$ має **лінійний потік інформації**:

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Градієнт по $C_t$:**

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t + \text{члени від } \frac{\partial f_t}{\partial C_{t-1}}, \frac{\partial i_t}{\partial C_{t-1}}$$

Якщо forget gate навчиться встановлювати $f_t \approx 1$ для важливої інформації, то:

$$\frac{\partial C_t}{\partial C_{t-1}} \approx 1$$

Це означає, що градієнт **не зменшується експоненційно** при зворотному поширенні через час!

**Порівняння з RNN:**

- **RNN:** $\frac{\partial h_t}{\partial h_{t-1}} = W_h^T \cdot \text{diag}(\tanh'(z_i))$ → експоненційне зменшення
- **LSTM:** $\frac{\partial C_t}{\partial C_{t-1}} \approx f_t$ → може бути близько до 1

### Математика сигмоїди та гіперболічного тангенсу

**Сигмоїда:** $\sigma(z) = \frac{1}{1 + e^{-z}}$

- Діапазон: $(0, 1)$
- Похідна: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- Використання: Gates (контроль потоку)

**Гіперболічний тангенс:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

- Діапазон: $(-1, 1)$
- Похідна: $\tanh'(z) = 1 - \tanh^2(z)$
- Використання: Трансформація значень (не лише контроль)

**Чому різні функції?**

- **Gates використовують $\sigma$:** Потрібен діапазон $[0, 1]$ для множення (скільки пропустити)
- **Значення використовують $\tanh$:** Потрібен діапазон $[-1, 1]$ для представлення як позитивних, так і негативних значень

## Інженерна інтерпретація

### Як LSTM "вирішує" забути старий тренд і запам'ятати новий інцидент

**Сценарій:** Аналіз метрик CPU для виявлення аномалій.

**Крок 1-100:** Нормальна робота (CPU ~ 30%)
- Forget gate: $f_t \approx 1$ (зберігаємо інформацію про нормальний стан)
- Input gate: $i_t \approx 0$ (немає нової важливої інформації)

**Крок 101:** Початок сплеску (CPU = 50%)
- Forget gate: $f_t \approx 0.8$ (частково "забуваємо" старий нормальний стан)
- Input gate: $i_t \approx 0.9$ (активно запам'ятовуємо новий сплеск)
- Cell state оновлюється: $C_t = 0.8 \cdot C_{t-1} + 0.9 \cdot \text{нове значення}$

**Крок 102-150:** Продовження сплеску (CPU = 70-80%)
- Forget gate: $f_t \approx 0.9$ (зберігаємо інформацію про сплеск)
- Input gate: $i_t \approx 0.7$ (оновлюємо інформацію про поточний рівень)

**Крок 151:** Повернення до норми (CPU = 30%)
- Forget gate: $f_t \approx 0.5$ (частково "забуваємо" інформацію про сплеск)
- Input gate: $i_t \approx 0.8$ (запам'ятовуємо повернення до норми)

**Результат:** LSTM динамічно адаптується, зберігаючи релевантну інформацію та "забуваючи" застарілу.

## Реалізація на Python

### Реалізація LSTM з нуля (для розуміння)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class LSTMCellFromScratch(nn.Module):
    """
    LSTM комірка, реалізована з нуля для демонстрації внутрішньої роботи.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCellFromScratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Ваги для forget gate
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size))
        self.b_f = nn.Parameter(torch.randn(hidden_size))
        
        # Ваги для input gate
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size))
        self.b_i = nn.Parameter(torch.randn(hidden_size))
        
        # Ваги для candidate values
        self.W_C = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size))
        self.b_C = nn.Parameter(torch.randn(hidden_size))
        
        # Ваги для output gate
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size + hidden_size))
        self.b_o = nn.Parameter(torch.randn(hidden_size))
        
        # Ініціалізація ваг
        self._init_weights()
    
    def _init_weights(self):
        """Ініціалізація ваг для стабільного навчання."""
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, 
                h_prev: torch.Tensor, 
                C_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass LSTM комірки.
        
        Parameters:
        -----------
        x : torch.Tensor
            Вхід на поточному кроці (batch_size, input_size)
        h_prev : torch.Tensor
            Попередній hidden state (batch_size, hidden_size)
        C_prev : torch.Tensor
            Попередній cell state (batch_size, hidden_size)
        
        Returns:
        --------
        h_t : torch.Tensor
            Новий hidden state
        C_t : torch.Tensor
            Новий cell state
        """
        # Конкатенація входу та попереднього hidden state
        combined = torch.cat([x, h_prev], dim=1)  # (batch_size, input_size + hidden_size)
        
        # Forget gate
        f_t = torch.sigmoid(combined @ self.W_f.T + self.b_f)
        
        # Input gate
        i_t = torch.sigmoid(combined @ self.W_i.T + self.b_i)
        
        # Candidate values
        C_tilde = torch.tanh(combined @ self.W_C.T + self.b_C)
        
        # Оновлення cell state
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Output gate
        o_t = torch.sigmoid(combined @ self.W_o.T + self.b_o)
        
        # Оновлення hidden state
        h_t = o_t * torch.tanh(C_t)
        
        return h_t, C_t

class LSTMFromScratch(nn.Module):
    """
    Повна LSTM мережа, побудована з комірок.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super(LSTMFromScratch, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            LSTMCellFromScratch(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass через послідовність.
        
        Parameters:
        -----------
        x : torch.Tensor
            Вхідна послідовність (batch_size, seq_length, input_size)
        
        Returns:
        --------
        output : torch.Tensor
            Вихід на останньому кроці (batch_size, hidden_size)
        """
        batch_size, seq_length, input_size = x.shape
        
        # Ініціалізація станів
        h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        C = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        
        # Прохід через послідовність
        for t in range(seq_length):
            x_t = x[:, t, :]  # (batch_size, input_size)
            
            for layer in range(self.num_layers):
                h[layer], C[layer] = self.cells[layer](
                    x_t if layer == 0 else h[layer-1],
                    h[layer],
                    C[layer]
                )
                x_t = h[layer]  # Для наступного шару
        
        return h[-1]  # Повертаємо hidden state останнього шару
```

### Візуалізація роботи gates

```python
def visualize_gates_activity(model: nn.Module, 
                              X_sample: torch.Tensor,
                              cell_idx: int = 0) -> dict:
    """
    Візуалізує активність gates під час forward pass.
    """
    model.eval()
    cell = model.cells[cell_idx]
    
    batch_size, seq_length, input_size = X_sample.shape
    hidden_size = cell.hidden_size
    
    # Ініціалізація
    h = torch.zeros(batch_size, hidden_size)
    C = torch.zeros(batch_size, hidden_size)
    
    # Зберігаємо значення gates для кожного кроку
    forget_gates = []
    input_gates = []
    output_gates = []
    cell_states = []
    
    with torch.no_grad():
        for t in range(seq_length):
            x_t = X_sample[:, t, :]
            combined = torch.cat([x_t, h], dim=1)
            
            # Обчислюємо gates
            f_t = torch.sigmoid(combined @ cell.W_f.T + cell.b_f)
            i_t = torch.sigmoid(combined @ cell.W_i.T + cell.b_i)
            o_t = torch.sigmoid(combined @ cell.W_o.T + cell.b_o)
            C_tilde = torch.tanh(combined @ cell.W_C.T + cell.b_C)
            
            # Оновлюємо стани
            C = f_t * C + i_t * C_tilde
            h = o_t * torch.tanh(C)
            
            # Зберігаємо значення (середнє по batch)
            forget_gates.append(f_t.mean(dim=0).cpu().numpy())
            input_gates.append(i_t.mean(dim=0).cpu().numpy())
            output_gates.append(o_t.mean(dim=0).cpu().numpy())
            cell_states.append(C.mean(dim=0).cpu().numpy())
    
    return {
        'forget_gates': np.array(forget_gates),
        'input_gates': np.array(input_gates),
        'output_gates': np.array(output_gates),
        'cell_states': np.array(cell_states)
    }

# Генерація тестового датасету зі сплеском
def generate_spike_dataset(n_samples: int = 100, seq_length: int = 200):
    """Генерує датасет з нормальною роботою та сплеском."""
    X = []
    y = []
    
    for _ in range(n_samples):
        sequence = np.random.normal(0.3, 0.1, seq_length)  # Нормальна робота
        
        # Додаємо сплеск на позиції 100-120
        spike_start = 100
        spike_end = 120
        sequence[spike_start:spike_end] = np.random.normal(0.8, 0.1, spike_end - spike_start)
        
        X.append(sequence.reshape(-1, 1))
        y.append(1.0 if np.max(sequence[spike_start:spike_end]) > 0.7 else 0.0)
    
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

# Генерація та візуалізація
X_gates, y_gates = generate_spike_dataset(n_samples=10, seq_length=200)
model_gates = LSTMFromScratch(input_size=1, hidden_size=5, num_layers=1)

# Навчання моделі (спрощене)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_gates.parameters(), lr=0.01)

for epoch in range(50):
    optimizer.zero_grad()
    outputs = model_gates(X_gates)
    loss = criterion(outputs, y_gates)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Візуалізація gates
gates_data = visualize_gates_activity(model_gates, X_gates[:1], cell_idx=0)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Вхідні дані
axes[0].plot(X_gates[0, :, 0].numpy(), linewidth=2, color='black')
axes[0].axvspan(100, 120, alpha=0.3, color='red', label='Сплеск')
axes[0].set_ylabel('Вхід', fontsize=12)
axes[0].set_title('Вхідна послідовність зі сплеском', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Forget gate
for i in range(gates_data['forget_gates'].shape[1]):
    axes[1].plot(gates_data['forget_gates'][:, i], alpha=0.6, linewidth=1)
axes[1].axvspan(100, 120, alpha=0.3, color='red')
axes[1].set_ylabel('Forget Gate', fontsize=12)
axes[1].set_title('Активність Forget Gate (низьке = забуваємо, високе = зберігаємо)', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Input gate
for i in range(gates_data['input_gates'].shape[1]):
    axes[2].plot(gates_data['input_gates'][:, i], alpha=0.6, linewidth=1)
axes[2].axvspan(100, 120, alpha=0.3, color='red')
axes[2].set_ylabel('Input Gate', fontsize=12)
axes[2].set_title('Активність Input Gate (високе = запам\'ятовуємо)', fontsize=14)
axes[2].grid(True, alpha=0.3)

# Cell state
for i in range(gates_data['cell_states'].shape[1]):
    axes[3].plot(gates_data['cell_states'][:, i], alpha=0.6, linewidth=1)
axes[3].axvspan(100, 120, alpha=0.3, color='red')
axes[3].set_xlabel('Крок $t$', fontsize=12)
axes[3].set_ylabel('Cell State', fontsize=12)
axes[3].set_title('Cell State (накопичена інформація)', fontsize=14)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_gates_activity.png', dpi=300)
plt.show()
```

### Порівняння LSTM з PyTorch та реалізацією з нуля

```python
class SimpleLSTMPyTorch(nn.Module):
    """LSTM використовуючи стандартний PyTorch модуль."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleLSTMPyTorch, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Порівняння на задачі довгострокової залежності
def generate_long_term_dependency_dataset(n_samples: int = 1000, 
                                          seq_length: int = 100,
                                          delay: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Генерує датасет, де правильна відповідь залежить від значення delay кроків тому.
    
    Задача: Визначити, чи було значення > 0.5 на кроці t-delay.
    Якщо так, то цільове значення = 1, інакше = 0.
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        sequence = np.random.rand(seq_length, 1)
        if sequence[seq_length - delay - 1, 0] > 0.5:
            target = 1.0
        else:
            target = 0.0
        X.append(sequence)
        y.append(target)
    
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

X_train, y_train = generate_long_term_dependency_dataset(n_samples=500, 
                                                          seq_length=100, 
                                                          delay=50)
X_test, y_test = generate_long_term_dependency_dataset(n_samples=100, 
                                                        seq_length=100, 
                                                        delay=50)

# Навчання PyTorch LSTM
model_pytorch = SimpleLSTMPyTorch(input_size=1, hidden_size=10, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_pytorch.parameters(), lr=0.001)

pytorch_losses = []
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model_pytorch(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_pytorch.parameters(), max_norm=1.0)
    optimizer.step()
    
    with torch.no_grad():
        test_outputs = model_pytorch(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        pytorch_losses.append(test_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Test Loss: {test_loss:.6f}")

# Навчання LSTM з нуля
model_scratch = LSTMFromScratch(input_size=1, hidden_size=10, num_layers=1)
fc_layer = nn.Linear(10, 1)
optimizer_scratch = torch.optim.Adam(
    list(model_scratch.parameters()) + list(fc_layer.parameters()), 
    lr=0.001
)

scratch_losses = []
for epoch in range(100):
    optimizer_scratch.zero_grad()
    hidden = model_scratch(X_train)
    outputs = fc_layer(hidden)
    loss = criterion(outputs, y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model_scratch.parameters()) + list(fc_layer.parameters()), 
        max_norm=1.0
    )
    optimizer_scratch.step()
    
    with torch.no_grad():
        test_hidden = model_scratch(X_test)
        test_outputs = fc_layer(test_hidden)
        test_loss = criterion(test_outputs, y_test).item()
        scratch_losses.append(test_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Test Loss: {test_loss:.6f}")

# Порівняння
plt.figure(figsize=(12, 6))
plt.plot(pytorch_losses, label='PyTorch LSTM', linewidth=2)
plt.plot(scratch_losses, label='LSTM з нуля', linewidth=2)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Test Loss', fontsize=14)
plt.title('Порівняння PyTorch LSTM та реалізації з нуля', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pytorch_vs_scratch_lstm.png', dpi=300)
plt.show()
```

### Аналіз градієнтів у LSTM

```python
def analyze_lstm_gradients(model: nn.Module, 
                           X_sample: torch.Tensor,
                           y_sample: torch.Tensor) -> dict:
    """
    Аналізує градієнти в LSTM для різних кроків послідовності.
    """
    model.train()
    criterion = nn.MSELoss()
    
    # Forward pass
    output = model(X_sample)
    loss = criterion(output, y_sample)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Збираємо градієнти з cell state
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.data.norm(2).item()
    
    return gradients

# Аналіз для послідовностей різної довжини
gradient_analysis = {}

for seq_len in [10, 20, 50, 100]:
    X_temp, y_temp = generate_long_term_dependency_dataset(
        n_samples=10, seq_length=seq_len, delay=seq_len//2
    )
    
    model_temp = SimpleLSTMPyTorch(input_size=1, hidden_size=10, output_size=1)
    grads = analyze_lstm_gradients(model_temp, X_temp, y_temp)
    
    # Середня норма градієнта
    avg_grad = np.mean(list(grads.values()))
    gradient_analysis[seq_len] = avg_grad

# Візуалізація
seq_lengths = list(gradient_analysis.keys())
grad_norms = list(gradient_analysis.values())

plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, grad_norms, 'o-', linewidth=2, markersize=10)
plt.xlabel('Довжина послідовності', fontsize=14)
plt.ylabel('Середня норма градієнта', fontsize=14)
plt.title('Градієнти в LSTM для різних довжин послідовностей', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_gradient_analysis.png', dpi=300)
plt.show()

print("\n" + "="*60)
print("Аналіз градієнтів LSTM:")
print("="*60)
for seq_len, grad_norm in gradient_analysis.items():
    print(f"Довжина {seq_len:3d}: норма градієнта = {grad_norm:.6f}")
print("\nЯкщо норми не зменшуються експоненційно, LSTM успішно обходить")
print("проблему зникаючого градієнта!")
```

## Висновки та наступні кроки

Ключові висновки:

1. **LSTM вирішує проблему зникаючого градієнта** через лінійний потік інформації в cell state
2. **Три gates** (Forget, Input, Output) дозволяють мережі активно керувати потоком інформації
3. **Cell state** слугує "конвеєрною стрічкою" для довгострокової пам'яті
4. **Hidden state** використовується для короткострокових виходів та контролю gates

**Для SRE практики:**
- LSTM може ефективно працювати з довгими послідовностями (логи за тиждень)
- Gates дозволяють мережі адаптивно "забувати" застарілу інформацію та "запам'ятовувати" нову
- LSTM підходить для предиктивного моніторингу, де важливі довгострокові залежності

**Обмеження LSTM:**
- Все ще вимагає послідовного проходження через весь ланцюг часу
- Інформація може "розмиватися" при довгих послідовностях
- Не може безпосередньо "стрибати" до важливих подій в минулому

У наступній лекції ми розглянемо **[механізм Attention](04b_attention_mechanism.md)**, який дозволяє моделі безпосередньо звертатися до будь-якого моменту в історії, не проходячи через всі проміжні кроки. Це особливо важливо для дуже довгих послідовностей, де LSTM може втрачати важливу інформацію через послідовну обробку. Після цього ми перейдемо до використання LSTM та Attention для **аномалійного виявлення** та **предиктивного моніторингу** в реальних IT-системах.

---

## Пов'язані теми

- **[Механізм Attention: Фокус на важливому без рекурсії](04b_attention_mechanism.md)** — розширення LSTM для роботи з дуже довгими послідовностями через прямий доступ до будь-якого моменту в історії

---

## Додаткові матеріали

### Рекомендована література

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural computation*, 9(8), 1735-1780.
2. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). "Learning to forget: Continual prediction with LSTM." *Neural computation*, 12(10), 2451-2471.
3. Olah, C. (2015). "Understanding LSTM Networks." *colah's blog*.

### Вправи для самостійної роботи

1. **Завдання 1:** Реалізуйте **Peephole Connections** в LSTM (додавання $C_{t-1}$ до входу gates). Порівняйте з класичною LSTM на задачі довгострокової залежності.

2. **Завдання 2:** Дослідіть вплив **ініціалізації forget gate bias** на навчання. Покажіть, що встановлення $b_f > 0$ допомагає мережі краще зберігати інформацію на початку навчання.

3. **Завдання 3:** Створіть **візуалізацію** роботи LSTM на реальних метриках (CPU, memory, latency). Покажіть, як gates реагують на різні типи подій (сплески, тренди, сезонність).

