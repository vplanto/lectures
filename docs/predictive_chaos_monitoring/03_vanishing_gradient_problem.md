---
title: "Чому RNN забувають історію"
layout: default
nav_order: 3
parent: "Блок 2: Архітектура Пам'яті (Deep Learning)"
---

# Чому RNN забувають історію

## Парадокс: Чому мережа, створена для пам'яті, забуває

Уявіть задачу: навчити нейронну мережу передбачати падіння сервера на основі логів за тиждень. Ви використовуєте RNN (Recurrent Neural Network) — архітектуру, спеціально розроблену для роботи з послідовностями. Але після навчання виявляється, що мережа "забуває" події, що сталися більше ніж 10 хвилин тому, навіть якщо вони критичні для прогнозу.

Це не баг — це фундаментальна математична проблема, яка називається **проблемою зникаючого градієнта** (vanishing gradient problem). І вона пояснює, чому класичні RNN не можуть навчитися довгостроковим залежностям.

## Математичний фундамент

### Архітектура RNN

**Стандартна RNN комірка:**

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

$$y_t = W_y h_t + b_y$$

Де:
- $h_t \in \mathbb{R}^d$ — прихований стан на кроці $t$
- $x_t \in \mathbb{R}^m$ — вхід на кроці $t$
- $W_h, W_x, W_y$ — матриці ваг
- $b, b_y$ — зміщення (bias)

**Функція втрат для послідовності довжини $T$:**

$$L = \sum_{t=1}^{T} \ell(y_t, \hat{y}_t)$$

Де $\ell$ — функція втрат (наприклад, MSE або cross-entropy).

### Backpropagation Through Time (BPTT)

Для навчання RNN використовується **Backpropagation Through Time** — розширення звичайного backpropagation для послідовностей.

**Градієнт по параметрам $W_h$:**

$$\frac{\partial L}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial L}{\partial y_t} \frac{\partial y_t}{\partial h_t} \frac{\partial h_t}{\partial W_h}$$

**Ключова складність:** $\frac{\partial h_t}{\partial W_h}$ залежить від всіх попередніх станів через ланцюгове правило.

### Ланцюгове правило та проблема зникаючого градієнта

**Розкладемо $\frac{\partial h_t}{\partial h_k}$ для $k < t$:**

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

**Обчислимо $\frac{\partial h_i}{\partial h_{i-1}}$:**

$$h_i = \tanh(W_h h_{i-1} + W_x x_i + b)$$

$$\frac{\partial h_i}{\partial h_{i-1}} = W_h^T \cdot \text{diag}(\tanh'(z_i))$$

Де $z_i = W_h h_{i-1} + W_x x_i + b$ та $\tanh'(z) = 1 - \tanh^2(z)$.

**Ключове спостереження:**

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} W_h^T \cdot \text{diag}(\tanh'(z_i))$$

**Оцінка норми:**

$$\left\|\frac{\partial h_t}{\partial h_k}\right\| \leq \|W_h^T\|^t \cdot \|\text{diag}(\tanh'(z_i))\|^t$$

Оскільки $\tanh'(z) \in (0, 1]$ (максимум 1 при $z=0$), маємо:

$$\left\|\frac{\partial h_t}{\partial h_k}\right\| \leq \|W_h\|^t$$

**Якщо $\|W_h\| < 1$** (спектральна норма менше 1), то:

$$\left\|\frac{\partial h_t}{\partial h_k}\right\| \to 0 \text{ експоненційно при } t - k \to \infty$$

Це означає, що градієнт **експоненційно зменшується** при зворотному поширенні через час.

### Градієнт по параметрам на відстані

**Градієнт по $W_h$ від помилки на кроці $t$:**

$$\frac{\partial L_t}{\partial W_h} = \frac{\partial L_t}{\partial y_t} \frac{\partial y_t}{\partial h_t} \sum_{k=1}^{t} \left(\prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}\right) \frac{\partial h_k}{\partial W_h}$$

**Для довгих послідовностей ($t \gg k$):**

Якщо $\|W_h\| < 1$, то внесок від ранніх кроків ($k \ll t$) експоненційно малий:

$$\left\|\prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}\right\| \approx 0 \text{ для } t - k \gg 1$$

**Висновок:** RNN не може навчитися довгостроковим залежностям, бо градієнти від ранніх кроків "зникають" до того, як досягають параметрів.

### Проблема вибухаючого градієнта

**Якщо $\|W_h\| > 1$:**

$$\left\|\frac{\partial h_t}{\partial h_k}\right\| \to \infty \text{ експоненційно при } t - k \to \infty$$

Це призводить до **вибухаючого градієнта** (exploding gradient), що робить навчання нестабільним.

**Оптимальна умова:** $\|W_h\| \approx 1$ (на межі стабільності), але це важко підтримувати.

## Інженерна інтерпретація

### Чому це критично для моніторингу

**Сценарій:** Аналіз логів за тиждень для передбачення падіння сервера.

- **Крок 1 (7 днів тому):** Початок деградації бази даних
- **Крок 2-1000:** Нормальна робота
- **Крок 1001 (сьогодні):** Сплеск помилок

**Проблема:** RNN не може "запам'ятати" подію з кроку 1, бо градієнт від кроку 1001 до кроку 1 експоненційно малий.

**Результат:** Мережа навчається лише короткостроковим патернам (останні 10-20 кроків), ігноруючи критичну інформацію з минулого.

### Практичні наслідки

1. **Обмежена ефективна довжина пам'яті:** Класичні RNN ефективно "пам'ятають" лише 5-20 кроків
2. **Нестабільність навчання:** Вибухаючі градієнти призводять до NaN під час навчання
3. **Неефективне використання даних:** Більшість історичних даних ігноруються

## Реалізація на Python

### Реалізація простої RNN

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleRNN(nn.Module):
    """
    Проста RNN для демонстрації проблеми зникаючого градієнта.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)
        # Використовуємо останній вихід
        out = self.fc(out[:, -1, :])
        return out

# Параметри
input_size = 1
hidden_size = 10
output_size = 1
seq_length = 100  # Довжина послідовності
```

### Генерація тестового датасету з довгостроковою залежністю

```python
def generate_long_term_dependency_dataset(n_samples: int = 1000, 
                                          seq_length: int = 100,
                                          delay: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Генерує датасет, де правильна відповідь залежить від значення delay кроків тому.
    
    Задача: Визначити, чи було значення > 0.5 на кроці t-delay.
    Якщо так, то цільове значення = 1, інакше = 0.
    
    Це тест на довгострокову пам'ять.
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Генеруємо випадкову послідовність
        sequence = np.random.rand(seq_length, 1)
        
        # Визначаємо ціль на основі значення delay кроків тому
        if sequence[seq_length - delay - 1, 0] > 0.5:
            target = 1.0
        else:
            target = 0.0
        
        X.append(sequence)
        y.append(target)
    
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

# Генерація датасету
X_train, y_train = generate_long_term_dependency_dataset(n_samples=500, 
                                                         seq_length=100, 
                                                         delay=50)
X_test, y_test = generate_long_term_dependency_dataset(n_samples=100, 
                                                       seq_length=100, 
                                                       delay=50)

print(f"Розмір навчального набору: {X_train.shape}")
print(f"Розмір тестового набору: {X_test.shape}")
```

### Навчання RNN та аналіз градієнтів

```python
def train_and_analyze_gradients(model: nn.Module, 
                                X_train: torch.Tensor, 
                                y_train: torch.Tensor,
                                X_test: torch.Tensor,
                                y_test: torch.Tensor,
                                epochs: int = 50) -> dict:
    """
    Навчає модель та аналізує градієнти для різних кроків послідовності.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    gradient_norms = []
    
    for epoch in range(epochs):
        # Навчання
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        
        # Збір норм градієнтів
        total_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        gradient_norms.append(total_norm)
        
        # Gradient clipping для запобігання вибуху
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
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Grad Norm: {total_norm:.4f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'gradient_norms': gradient_norms
    }

# Створення та навчання моделі
model_rnn = SimpleRNN(input_size=1, hidden_size=10, output_size=1)
results = train_and_analyze_gradients(model_rnn, X_train, y_train, X_test, y_test, epochs=100)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(results['train_losses'], label='Train Loss', linewidth=2)
axes[0].plot(results['test_losses'], label='Test Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Навчання RNN на довгостроковій залежності', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(results['gradient_norms'], linewidth=2, color='red')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Норма градієнта', fontsize=12)
axes[1].set_title('Норма градієнта під час навчання', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_training.png', dpi=300)
plt.show()
```

### Візуалізація градієнтів для різних кроків послідовності

```python
def analyze_gradient_flow(model: nn.Module, 
                          X_sample: torch.Tensor, 
                          y_sample: torch.Tensor,
                          seq_length: int) -> np.ndarray:
    """
    Аналізує, як градієнти змінюються для різних кроків послідовності.
    """
    model.eval()
    criterion = nn.MSELoss()
    
    # Створюємо послідовності різної довжини
    gradient_contributions = []
    
    for t in range(10, seq_length + 1, 10):
        # Беремо перші t кроків
        X_truncated = X_sample[:, :t, :]
        
        # Forward pass
        output = model(X_truncated)
        loss = criterion(output, y_sample)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Збираємо градієнти з RNN шару
        total_grad_norm = 0
        for name, param in model.named_parameters():
            if 'rnn' in name and param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        
        gradient_contributions.append(np.sqrt(total_grad_norm))
    
    return np.array(gradient_contributions)

# Аналіз для одного прикладу
X_sample = X_test[:1]
y_sample = y_test[:1]

gradient_flow = analyze_gradient_flow(model_rnn, X_sample, y_sample, seq_length=100)

plt.figure(figsize=(10, 6))
steps = np.arange(10, 101, 10)
plt.plot(steps, gradient_flow, 'o-', linewidth=2, markersize=8)
plt.xlabel('Довжина послідовності (кроків)', fontsize=14)
plt.ylabel('Норма градієнта', fontsize=14)
plt.title('Зміна градієнта в залежності від довжини послідовності', fontsize=16)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('gradient_flow.png', dpi=300)
plt.show()

print(f"\nГрадієнт для послідовності довжиною 10: {gradient_flow[0]:.6f}")
print(f"Градієнт для послідовності довжиною 100: {gradient_flow[-1]:.6f}")
print(f"Відношення: {gradient_flow[-1] / gradient_flow[0]:.6f}")
print("\nЯкщо відношення << 1, це підтверджує проблему зникаючого градієнта.")
```

### Демонстрація експоненційного зменшення градієнта

```python
def demonstrate_vanishing_gradient_theory():
    """
    Демонструє теоретичне експоненційне зменшення градієнта.
    """
    # Симулюємо градієнт для різних значень ||W_h||
    steps = np.arange(1, 101)
    
    # Різні значення спектральної норми
    norms = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05]
    
    plt.figure(figsize=(12, 8))
    
    for norm in norms:
        # Теоретичне зменшення: ||gradient|| ~ norm^t
        gradient_magnitude = norm ** steps
        plt.plot(steps, gradient_magnitude, linewidth=2, 
                label=f'||W_h|| = {norm}')
    
    plt.xlabel('Кількість кроків назад (t)', fontsize=14)
    plt.ylabel('Відносна величина градієнта', fontsize=14)
    plt.title('Теоретичне зменшення градієнта в RNN', fontsize=16)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1e-6, color='r', linestyle='--', linewidth=1, 
               label='Практичний поріг (1e-6)')
    plt.tight_layout()
    plt.savefig('vanishing_gradient_theory.png', dpi=300)
    plt.show()
    
    # Обчислюємо ефективну довжину пам'яті
    print("\n" + "="*60)
    print("Ефективна довжина пам'яті (градієнт > 1e-6):")
    print("="*60)
    for norm in norms:
        if norm < 1:
            effective_length = int(np.log(1e-6) / np.log(norm))
            print(f"||W_h|| = {norm:.2f}: ~{effective_length} кроків")
        else:
            print(f"||W_h|| = {norm:.2f}: необмежена (вибухає)")

demonstrate_vanishing_gradient_theory()
```

### Порівняння з LSTM (попередній огляд)

```python
class SimpleLSTM(nn.Module):
    """
    Проста LSTM для порівняння з RNN.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Порівняння RNN vs LSTM
print("\n" + "="*60)
print("Порівняння RNN vs LSTM на довгостроковій залежності")
print("="*60)

# Навчання LSTM
model_lstm = SimpleLSTM(input_size=1, hidden_size=10, output_size=1)
results_lstm = train_and_analyze_gradients(model_lstm, X_train, y_train, 
                                           X_test, y_test, epochs=100)

# Порівняння фінальних помилок
model_rnn.eval()
model_lstm.eval()
with torch.no_grad():
    rnn_pred = model_rnn(X_test)
    lstm_pred = model_lstm(X_test)
    rnn_error = nn.MSELoss()(rnn_pred, y_test).item()
    lstm_error = nn.MSELoss()(lstm_pred, y_test).item()

print(f"\nФінальна помилка RNN: {rnn_error:.6f}")
print(f"Фінальна помилка LSTM: {lstm_error:.6f}")
print(f"Покращення: {rnn_error / lstm_error:.2f}x")

# Візуалізація порівняння
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(results['test_losses'], label='RNN', linewidth=2)
axes[0].plot(results_lstm['test_losses'], label='LSTM', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Test Loss', fontsize=12)
axes[0].set_title('Порівняння навчання RNN vs LSTM', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(results['gradient_norms'], label='RNN', linewidth=2)
axes[1].plot(results_lstm['gradient_norms'], label='LSTM', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Норма градієнта', fontsize=12)
axes[1].set_title('Норма градієнта: RNN vs LSTM', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_vs_lstm.png', dpi=300)
plt.show()
```

## Висновки та наступні кроки

Ключові висновки:

1. **Проблема зникаючого градієнта** виникає через експоненційне зменшення градієнтів при зворотному поширенні через час
2. **Математична причина:** Ланцюгове правило призводить до добутку багатьох похідних, кожна з яких $< 1$
3. **Практичний ефект:** RNN ефективно "пам'ятають" лише 5-20 кроків, незважаючи на архітектуру для послідовностей
4. **Gradient clipping** допомагає з вибухаючими градієнтами, але не вирішує проблему зникаючих

**Для SRE практики:**
- Класичні RNN не підходять для аналізу довгих послідовностей (логи за тиждень)
- Потрібні архітектури, які обходять проблему зникаючого градієнта
- LSTM та GRU були розроблені саме для вирішення цієї проблеми

У наступній лекції ми розглянемо **[LSTM як "Гідравлічна система" управління інформацією](04_lstm_hydraulic_memory.md)** — архітектуру, яка вирішує проблему зникаючого градієнта через механізм Cell State, що забезпечує лінійний потік інформації та дозволяє зберігати довгострокову пам'ять.

---

## Пов'язані теми

- **[LSTM як "Гідравлічна система" управління інформацією](04_lstm_hydraulic_memory.md)** — архітектурне рішення проблеми зникаючого градієнта через Cell State та механізм gates

---

## Додаткові матеріали

### Рекомендована література

1. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural computation*, 9(8), 1735-1780.
2. Bengio, Y., Simard, P., & Frasconi, P. (1994). "Learning long-term dependencies with gradient descent is difficult." *IEEE transactions on neural networks*, 5(2), 157-166.
3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). "On the difficulty of training recurrent neural networks." *ICML*.

### Вправи для самостійної роботи

1. **Завдання 1:** Реалізуйте **GRU (Gated Recurrent Unit)** та порівняйте його з RNN та LSTM на задачі довгострокової залежності. Проаналізуйте градієнти для всіх трьох архітектур.

2. **Завдання 2:** Дослідіть вплив **ініціалізації ваг** (Xavier, He) на проблему зникаючого градієнта. Покажіть, як правильна ініціалізація може покращити навчання RNN.

3. **Завдання 3:** Реалізуйте **Residual Connections** в RNN (ResRNN) та порівняйте з класичною RNN. Чи допомагають skip connections з проблемою зникаючого градієнта?

