# 11_uncertainty_weighted_loss_seminar.md: Семінар з Uncertainty-Weighted Loss

**Курс:** Геометричний Deep Learning в Логістиці
**Модуль:** Додатковий семінар
**Рівень:** Advanced / Expert
**Тривалість:** 2-3 години

---

## 1. Вступ: Проблема балансування Loss компонентів

### 1.1. Класична проблема: "Підбір на око"

**Сценарій:** Ви навчаєте модель для VRP з багатьма компонентами Loss:

$$\mathcal{L}_{total} = \alpha \mathcal{L}_{length} + \beta \mathcal{L}_{perm} + \gamma \mathcal{L}_{time} + \delta \mathcal{L}_{capacity}$$

**Проблема:** Як вибрати ваги $\alpha, \beta, \gamma, \delta$?

**Типовий підхід (неправильний):**
```python
# Спробуємо різні комбінації
alpha, beta, gamma, delta = 1.0, 0.1, 1.0, 1.0  # Не працює
alpha, beta, gamma, delta = 0.1, 0.1, 10.0, 5.0  # Теж не працює
alpha, beta, gamma, delta = 0.01, 0.1, 100.0, 50.0  # Може працювати?
# ... 50+ ітерацій підбору ...
```

**Чому це погано:**
1. **Відсутність теоретичного обґрунтування:** Ваги обираються емпірично
2. **Нестабільність:** Ваги, які працюють на тренувальному сеті, можуть не працювати на валідаційному
3. **Втрата часу:** Тижні експериментів замість автоматичного балансування
4. **Неадаптивність:** Ваги не змінюються під час навчання

### 1.2. Математична формалізація проблеми

**Градієнт загального Loss:**

$$\frac{\partial \mathcal{L}_{total}}{\partial \theta} = \alpha \frac{\partial \mathcal{L}_{length}}{\partial \theta} + \beta \frac{\partial \mathcal{L}_{perm}}{\partial \theta} + \gamma \frac{\partial \mathcal{L}_{time}}{\partial \theta} + \delta \frac{\partial \mathcal{L}_{capacity}}{\partial \theta}$$

**Проблема домінування:**

Якщо норми градієнтів сильно різняться:
- $\|\alpha \frac{\partial \mathcal{L}_{length}}{\partial \theta}\| = 1000$
- $\|\gamma \frac{\partial \mathcal{L}_{time}}{\partial \theta}\| = 1$

То градієнт домінується першим компонентом, і модель ігнорує другий.

**Мета:** Автоматично балансувати ваги так, щоб кожен компонент мав рівний вплив на оновлення параметрів.

---

## 2. Uncertainty-Weighted Loss: Математичне обґрунтування

### 2.1. Ідея: Моделювання невизначеності

**Ключова інтуїція:** Замість фіксованих ваг, введемо **learnable параметри невизначеності** $\sigma_i$ для кожного компонента Loss.

**Гіпотеза:** Якщо модель "впевнена" в компоненті $i$ (низька невизначеність $\sigma_i$), то цей компонент має мати більший вплив. Якщо модель "не впевнена" (висока невизначеність $\sigma_i$), то компонент має мати менший вплив.

### 2.2. Виведення з байєсівської оптимізації

**Крок 1: Моделювання невизначеності через гаусівський шум**

Припустимо, що кожен компонент Loss має **гаусівську невизначеність**:

$$\mathcal{L}_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$$

Де:
- $\mu_i$ — спостережене значення Loss компонента $i$
- $\sigma_i$ — невизначеність (стандартне відхилення)

**Крок 2: Negative Log-Likelihood (NLL)**

Для гаусівського розподілу, negative log-likelihood:

$$-\log p(\mathcal{L}_i | \mu_i, \sigma_i) = \frac{1}{2\sigma_i^2}(\mathcal{L}_i - \mu_i)^2 + \frac{1}{2}\log(2\pi\sigma_i^2)$$

**Спрощення:** Припускаємо, що $\mu_i = 0$ (мінімізуємо Loss, тому очікуване значення = 0):

$$-\log p(\mathcal{L}_i | \sigma_i) = \frac{1}{2\sigma_i^2}\mathcal{L}_i^2 + \frac{1}{2}\log(2\pi\sigma_i^2)$$

**Крок 3: Сума по всіх компонентах**

Для незалежних компонентів:

$$-\log p(\mathcal{L}_{total} | \{\sigma_i\}) = \sum_{i} \left[\frac{1}{2\sigma_i^2}\mathcal{L}_i^2 + \frac{1}{2}\log(2\pi\sigma_i^2)\right]$$

**Крок 4: Спрощення константи**

Виносимо константу $\frac{1}{2}\log(2\pi)$:

$$-\log p(\mathcal{L}_{total} | \{\sigma_i\}) = \sum_{i} \left[\frac{1}{2\sigma_i^2}\mathcal{L}_i^2 + \log \sigma_i\right] + \text{const}$$

**Крок 5: Фінальна формула Uncertainty-Weighted Loss**

$$\mathcal{L}_{total} = \sum_{i} \left[\frac{1}{2\sigma_i^2}\mathcal{L}_i + \log \sigma_i\right]$$

Де:
- $\frac{1}{\sigma_i^2}$ — автоматична вага для компонента $i$
- $\log \sigma_i$ — regularization термін (штрафує за надто великі $\sigma_i$)

### 2.3. Альтернативне виведення: Мінімізація вариації

**Альтернативний підхід:** Мінімізувати **вариансу** загального Loss при обмеженні на середнє значення.

**Задача оптимізації:**

$$\min_{\{\sigma_i\}} \text{Var}\left[\sum_{i} w_i \mathcal{L}_i\right]$$

За обмеженням:
$$\mathbb{E}\left[\sum_{i} w_i \mathcal{L}_i\right] = \text{const}$$

**Рішення через метод Лагранжа:**

Вводимо множник Лагранжа $\lambda$:

$$\mathcal{L} = \text{Var}\left[\sum_{i} w_i \mathcal{L}_i\right] + \lambda \left(\mathbb{E}\left[\sum_{i} w_i \mathcal{L}_i\right] - \text{const}\right)$$

**Якщо припустити незалежність компонентів:**

$$\text{Var}\left[\sum_{i} w_i \mathcal{L}_i\right] = \sum_{i} w_i^2 \text{Var}[\mathcal{L}_i] = \sum_{i} w_i^2 \sigma_i^2$$

**Мінімізація:**

$$\frac{\partial \mathcal{L}}{\partial w_i} = 2w_i \sigma_i^2 + \lambda = 0$$

Отже:
$$w_i = -\frac{\lambda}{2\sigma_i^2}$$

**Нормалізація:** Якщо $\lambda = -2$, то $w_i = \frac{1}{\sigma_i^2}$, що збігається з нашим виведенням!

---

## 3. Детальний математичний аналіз

### 3.1. Інтерпретація параметрів $\sigma_i$

**Фізичний зміст:**

- **$\sigma_i$ малий** (наприклад, $\sigma_i = 0.1$):
  - $\frac{1}{\sigma_i^2} = 100$ — велика вага
  - Модель "впевнена" в цьому компоненті
  - Компонент має сильний вплив на оновлення параметрів

- **$\sigma_i$ великий** (наприклад, $\sigma_i = 10$):
  - $\frac{1}{\sigma_i^2} = 0.01$ — мала вага
  - Модель "не впевнена" в цьому компоненті
  - Компонент має слабкий вплив на оновлення параметрів

**Графічна інтерпретація:**

```
Вага w_i = 1/(2σ_i²)
    ↑
    |     ╱
    |    ╱
    |   ╱
    |  ╱
    | ╱
    |╱
    └──────────────→ σ_i
    0    1    2    3
```

### 3.2. Regularization термін $\log \sigma_i$

**Чому потрібен regularization термін?**

Без regularization, модель може зробити $\sigma_i$ дуже великим, щоб повністю ігнорувати компонент:

$$\lim_{\sigma_i \to \infty} \frac{1}{2\sigma_i^2}\mathcal{L}_i = 0$$

**Regularization термін штрафує за великі $\sigma_i$:**

$$\lim_{\sigma_i \to \infty} \log \sigma_i = \infty$$

**Баланс:**

Модель навчається знаходити оптимальний баланс між:
- **Мінімізацією Loss:** $\frac{1}{2\sigma_i^2}\mathcal{L}_i$ (зменшує $\sigma_i$)
- **Regularization:** $\log \sigma_i$ (збільшує $\sigma_i$)

### 3.3. Градієнти по $\sigma_i$

**Обчислення градієнта:**

$$\frac{\partial \mathcal{L}_{total}}{\partial \sigma_i} = \frac{\partial}{\partial \sigma_i}\left[\frac{1}{2\sigma_i^2}\mathcal{L}_i + \log \sigma_i\right]$$

**Крок 1:** Похідна першого терміну:

$$\frac{\partial}{\partial \sigma_i}\left[\frac{1}{2\sigma_i^2}\mathcal{L}_i\right] = -\frac{1}{\sigma_i^3}\mathcal{L}_i$$

**Крок 2:** Похідна другого терміну:

$$\frac{\partial}{\partial \sigma_i}\left[\log \sigma_i\right] = \frac{1}{\sigma_i}$$

**Фінальний градієнт:**

$$\frac{\partial \mathcal{L}_{total}}{\partial \sigma_i} = -\frac{1}{\sigma_i^3}\mathcal{L}_i + \frac{1}{\sigma_i} = \frac{1}{\sigma_i}\left(1 - \frac{\mathcal{L}_i}{\sigma_i^2}\right)$$

**Інтерпретація:**

- Якщо $\mathcal{L}_i > \sigma_i^2$: градієнт негативний → $\sigma_i$ зменшується → вага збільшується
- Якщо $\mathcal{L}_i < \sigma_i^2$: градієнт позитивний → $\sigma_i$ збільшується → вага зменшується

**Рівновага:**

Навчання зупиняється, коли:
$$\frac{\mathcal{L}_i}{\sigma_i^2} = 1 \quad \Rightarrow \quad \sigma_i^2 = \mathcal{L}_i$$

Тобто, $\sigma_i$ автоматично встановлюється рівним середньоквадратичному значенню $\mathcal{L}_i$!

### 3.4. Стабільність: Використання $\log \sigma_i$ замість $\sigma_i$

**Проблема:** Якщо навчати $\sigma_i$ напряму, він може стати негативним або дуже малим.

**Рішення:** Навчаємо $\log \sigma_i$ замість $\sigma_i$:

$$\sigma_i = \exp(\log \sigma_i)$$

**Переваги:**

1. **Автоматична позитивність:** $\exp(x) > 0$ для будь-якого $x$
2. **Числова стабільність:** Градієнти більш стабільні
3. **Логарифмічний масштаб:** Легше навчати параметри різних порядків

**Градієнт по $\log \sigma_i$:**

$$\frac{\partial \mathcal{L}_{total}}{\partial \log \sigma_i} = \frac{\partial \mathcal{L}_{total}}{\partial \sigma_i} \cdot \frac{\partial \sigma_i}{\partial \log \sigma_i} = \frac{\partial \mathcal{L}_{total}}{\partial \sigma_i} \cdot \sigma_i$$

Підставляючи наш попередній результат:

$$\frac{\partial \mathcal{L}_{total}}{\partial \log \sigma_i} = \frac{1}{\sigma_i}\left(1 - \frac{\mathcal{L}_i}{\sigma_i^2}\right) \cdot \sigma_i = 1 - \frac{\mathcal{L}_i}{\sigma_i^2}$$

**Спрощена форма:**

$$\frac{\partial \mathcal{L}_{total}}{\partial \log \sigma_i} = 1 - \frac{\mathcal{L}_i}{\sigma_i^2}$$

---

## 4. Практична реалізація

### 4.1. Базова реалізація в PyTorch

```python
import torch
import torch.nn as nn

class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-Weighted Loss для автоматичного балансування компонентів.
    
    Формула: L_total = sum_i [1/(2*σ_i²) * L_i + log(σ_i)]
    """
    def __init__(self, num_components=4, init_log_sigma=0.0):
        """
        Args:
            num_components: кількість компонентів Loss
            init_log_sigma: початкове значення log(σ_i) (зазвичай 0.0)
        """
        super().__init__()
        # Learnable параметри: log(σ_i) для кожного компонента
        self.log_sigmas = nn.Parameter(
            torch.full((num_components,), init_log_sigma)
        )
    
    def forward(self, losses):
        """
        Args:
            losses: dict або list з компонентами Loss
                   {'length': tensor, 'perm': tensor, ...}
                   або [tensor, tensor, ...]
        
        Returns:
            total_loss: скалярний тензор
        """
        # Конвертуємо dict в list, якщо потрібно
        if isinstance(losses, dict):
            loss_list = list(losses.values())
        else:
            loss_list = losses
        
        # Перевірка розмірності
        assert len(loss_list) == len(self.log_sigmas), \
            f"Кількість компонентів Loss ({len(loss_list)}) не співпадає з кількістю параметрів ({len(self.log_sigmas)})"
        
        # Обчислюємо σ_i = exp(log_sigma_i)
        sigmas = torch.exp(self.log_sigmas)
        
        # Обчислюємо загальний Loss
        total_loss = 0.0
        for i, loss in enumerate(loss_list):
            # Вага: 1/(2*σ_i²)
            weight = 1.0 / (2.0 * sigmas[i]**2 + 1e-8)  # +1e-8 для числової стабільності
            
            # Компонент: weight * L_i + log(σ_i)
            component_loss = weight * loss + self.log_sigmas[i]
            
            total_loss += component_loss
        
        return total_loss
    
    def get_weights(self):
        """
        Повертає поточні ваги 1/(2*σ_i²) для кожного компонента.
        """
        sigmas = torch.exp(self.log_sigmas)
        weights = 1.0 / (2.0 * sigmas**2 + 1e-8)
        return weights.detach().cpu().numpy()
    
    def get_sigmas(self):
        """
        Повертає поточні значення σ_i для кожного компонента.
        """
        return torch.exp(self.log_sigmas).detach().cpu().numpy()
```

### 4.2. Використання в циклі навчання

```python
# Ініціалізація
uncertainty_loss = UncertaintyWeightedLoss(num_components=4)

# Оптимізатор (важливо: включаємо log_sigmas!)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(uncertainty_loss.parameters()),
    lr=1e-3
)

# Цикл навчання
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch)
        
        # Обчислюємо компоненти Loss
        losses = {
            'length': compute_length_loss(predictions, batch),
            'perm': compute_perm_loss(predictions, batch),
            'time': compute_time_loss(predictions, batch),
            'capacity': compute_capacity_loss(predictions, batch)
        }
        
        # Обчислюємо загальний Loss з автоматичним балансуванням
        total_loss = uncertainty_loss(losses)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Моніторинг (опціонально)
        if step % 100 == 0:
            weights = uncertainty_loss.get_weights()
            sigmas = uncertainty_loss.get_sigmas()
            print(f"Step {step}:")
            print(f"  Weights: {weights}")
            print(f"  Sigmas: {sigmas}")
            print(f"  Loss components: {[l.item() for l in losses.values()]}")
```

### 4.3. Розширена реалізація з моніторингом

```python
class UncertaintyWeightedLossWithMonitoring(UncertaintyWeightedLoss):
    """
    Розширена версія з моніторингом та логуванням.
    """
    def __init__(self, num_components=4, init_log_sigma=0.0, 
                 component_names=None):
        super().__init__(num_components, init_log_sigma)
        self.component_names = component_names or [f"component_{i}" for i in range(num_components)]
        self.history = {
            'weights': [],
            'sigmas': [],
            'losses': []
        }
    
    def forward(self, losses):
        total_loss = super().forward(losses)
        
        # Зберігаємо історію
        with torch.no_grad():
            weights = self.get_weights()
            sigmas = self.get_sigmas()
            
            self.history['weights'].append(weights.copy())
            self.history['sigmas'].append(sigmas.copy())
            
            if isinstance(losses, dict):
                loss_values = [l.item() for l in losses.values()]
            else:
                loss_values = [l.item() for l in losses]
            self.history['losses'].append(loss_values)
        
        return total_loss
    
    def plot_history(self):
        """
        Візуалізує історію ваг та невизначеностей.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Графік ваг
        axes[0, 0].set_title('Weights over time')
        for i, name in enumerate(self.component_names):
            weights_history = [w[i] for w in self.history['weights']]
            axes[0, 0].plot(weights_history, label=name)
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Weight')
        axes[0, 0].set_yscale('log')
        
        # Графік σ_i
        axes[0, 1].set_title('Uncertainties (σ_i) over time')
        for i, name in enumerate(self.component_names):
            sigmas_history = [s[i] for s in self.history['sigmas']]
            axes[0, 1].plot(sigmas_history, label=name)
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('σ_i')
        
        # Графік Loss компонентів
        axes[1, 0].set_title('Loss components over time')
        for i, name in enumerate(self.component_names):
            losses_history = [l[i] for l in self.history['losses']]
            axes[1, 0].plot(losses_history, label=name)
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Loss')
        
        # Графік співвідношення Loss / σ²
        axes[1, 1].set_title('Loss / σ² ratio (should → 1)')
        for i, name in enumerate(self.component_names):
            ratios = []
            for step in range(len(self.history['losses'])):
                loss = self.history['losses'][step][i]
                sigma_sq = self.history['sigmas'][step][i]**2
                ratios.append(loss / (sigma_sq + 1e-8))
            axes[1, 1].plot(ratios, label=name)
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Target')
        axes[1, 1].legend()
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss / σ²')
        
        plt.tight_layout()
        return fig
```

---

## 5. Математичні вправи та докази

### 5.1. Вправа 1: Доведення оптимальності

**Задача:** Довести, що для фіксованих $\mathcal{L}_i$, оптимальне значення $\sigma_i$ дорівнює:

$$\sigma_i^* = \sqrt{\mathcal{L}_i}$$

**Рішення:**

Мінімізуємо функцію:

$$f(\sigma_i) = \frac{1}{2\sigma_i^2}\mathcal{L}_i + \log \sigma_i$$

**Крок 1:** Обчислюємо похідну:

$$f'(\sigma_i) = -\frac{1}{\sigma_i^3}\mathcal{L}_i + \frac{1}{\sigma_i} = \frac{1}{\sigma_i}\left(1 - \frac{\mathcal{L}_i}{\sigma_i^2}\right)$$

**Крок 2:** Прирівнюємо до нуля:

$$f'(\sigma_i) = 0 \quad \Rightarrow \quad 1 - \frac{\mathcal{L}_i}{\sigma_i^2} = 0$$

**Крок 3:** Розв'язуємо:

$$\sigma_i^2 = \mathcal{L}_i \quad \Rightarrow \quad \sigma_i^* = \sqrt{\mathcal{L}_i}$$

**Крок 4:** Перевіряємо другу похідну (мінімум):

$$f''(\sigma_i) = \frac{3\mathcal{L}_i}{\sigma_i^4} - \frac{1}{\sigma_i^2}$$

Підставляючи $\sigma_i^* = \sqrt{\mathcal{L}_i}$:

$$f''(\sigma_i^*) = \frac{3\mathcal{L}_i}{\mathcal{L}_i^2} - \frac{1}{\mathcal{L}_i} = \frac{2}{\mathcal{L}_i} > 0$$

Отже, це дійсно мінімум! ✅

### 5.2. Вправа 2: Асимптотична поведінка

**Задача:** Дослідити асимптотичну поведінку ваги $w_i = \frac{1}{2\sigma_i^2}$ при $\sigma_i \to 0$ та $\sigma_i \to \infty$.

**Рішення:**

**Випадок 1: $\sigma_i \to 0$**

$$\lim_{\sigma_i \to 0} w_i = \lim_{\sigma_i \to 0} \frac{1}{2\sigma_i^2} = \infty$$

**Інтерпретація:** Якщо невизначеність дуже мала, вага стає дуже великою. Але regularization термін $\log \sigma_i \to -\infty$ штрафує за це.

**Випадок 2: $\sigma_i \to \infty$**

$$\lim_{\sigma_i \to \infty} w_i = \lim_{\sigma_i \to \infty} \frac{1}{2\sigma_i^2} = 0$$

**Інтерпретація:** Якщо невизначеність дуже велика, вага стає нульовою. Але regularization термін $\log \sigma_i \to \infty$ штрафує за це.

**Висновок:** Regularization термін запобігає екстремальним значенням $\sigma_i$.

### 5.3. Вправа 3: Порівняння з фіксованими вагами

**Задача:** Показати, що Uncertainty-Weighted Loss еквівалентна фіксованим вагам, якщо $\sigma_i$ не навчаються.

**Рішення:**

Якщо $\sigma_i$ фіксовані (не learnable), то:

$$\mathcal{L}_{total} = \sum_{i} \left[\frac{1}{2\sigma_i^2}\mathcal{L}_i + \log \sigma_i\right] = \sum_{i} \frac{1}{2\sigma_i^2}\mathcal{L}_i + \text{const}$$

Де константа $\sum_i \log \sigma_i$ не впливає на градієнти.

**Висновок:** Фіксовані ваги — це окремий випадок Uncertainty-Weighted Loss з фіксованими $\sigma_i$.

### 5.4. Вправа 4: Інваріантність до масштабування

**Задача:** Показати, що Uncertainty-Weighted Loss інваріантна до масштабування компонентів Loss.

**Рішення:**

Нехай $\mathcal{L}_i' = k \mathcal{L}_i$ для деякого $k > 0$.

Тоді оптимальне $\sigma_i'$:

$$\sigma_i' = \sqrt{\mathcal{L}_i'} = \sqrt{k \mathcal{L}_i} = \sqrt{k} \sqrt{\mathcal{L}_i} = \sqrt{k} \sigma_i$$

**Вага:**

$$w_i' = \frac{1}{2(\sigma_i')^2} = \frac{1}{2k\sigma_i^2} = \frac{1}{k} w_i$$

**Внесок до Loss:**

$$w_i' \mathcal{L}_i' = \frac{1}{k} w_i \cdot k \mathcal{L}_i = w_i \mathcal{L}_i$$

**Висновок:** Масштабування компонентів Loss не змінює відносний вплив компонентів! ✅

---

## 6. Порівняння з альтернативними методами

### 6.1. Фіксовані ваги

**Формула:**
$$\mathcal{L}_{total} = \sum_{i} w_i \mathcal{L}_i$$

**Переваги:**
- Простота реалізації
- Швидкість обчислення

**Недоліки:**
- Потребує ручного підбору
- Не адаптується до змін під час навчання
- Нестабільність на різних датасетах

### 6.2. Gradient Norm Balancing

**Формула:**
$$w_i^{(t+1)} = w_i^{(t)} \cdot \frac{\bar{n}^{(t)}}{n_i^{(t)} + \epsilon}$$

Де $n_i^{(t)} = \|\frac{\partial \mathcal{L}_i}{\partial \theta}\|$ — норма градієнта.

**Переваги:**
- Автоматичне балансування
- Адаптується під час навчання

**Недоліки:**
- Потребує обчислення градієнтів для кожного компонента окремо
- Може бути нестабільним на початку навчання

### 6.3. Uncertainty-Weighted Loss

**Формула:**
$$\mathcal{L}_{total} = \sum_{i} \left[\frac{1}{2\sigma_i^2}\mathcal{L}_i + \log \sigma_i\right]$$

**Переваги:**
- ✅ **Теоретичне обґрунтування:** Виведено з байєсівської оптимізації
- ✅ **Повна автоматизація:** Не потребує ручного підбору
- ✅ **Адаптивність:** Автоматично адаптується під час навчання
- ✅ **Інваріантність:** Інваріантна до масштабування компонентів
- ✅ **Інтерпретація:** $\sigma_i$ має чіткий фізичний зміст (невизначеність)

**Недоліки:**
- Потребує додаткових параметрів (log_sigmas)
- Може бути повільнішим через додаткові обчислення

### 6.4. Таблиця порівняння

| Критерій | Фіксовані ваги | Gradient Norm | Uncertainty-Weighted |
|----------|----------------|---------------|---------------------|
| **Теоретичне обґрунтування** | ❌ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Автоматизація** | ❌ | ✅ | ✅ |
| **Адаптивність** | ❌ | ✅ | ✅ |
| **Складність реалізації** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Числова стабільність** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Інтерпретація** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Інваріантність до масштабу** | ❌ | ⭐⭐⭐ | ✅ |

---

## 7. Практичні рекомендації

### 7.1. Коли використовувати Uncertainty-Weighted Loss?

**Рекомендується:**

1. ✅ **Багатокомпонентний Loss:** Коли маєте 3+ компонентів з різними масштабами
2. ✅ **Нестабільне навчання:** Коли фіксовані ваги призводять до нестабільності
3. ✅ **Автоматизація:** Коли хочете уникнути ручного підбору ваг
4. ✅ **Різні швидкості збіжності:** Коли компоненти збігаються з різною швидкістю

**Не рекомендується:**

1. ❌ **Простий Loss:** Коли маєте 1-2 компоненти з однаковими масштабами
2. ❌ **Обмежені ресурси:** Коли додаткові параметри критичні для пам'яті
3. ❌ **Дуже малі датасети:** Коли недостатньо даних для навчання $\sigma_i$

### 7.2. Початкові значення $\log \sigma_i$

**Рекомендації:**

- **$\log \sigma_i = 0$** (тобто $\sigma_i = 1$): Стандартний вибір, починає з рівних ваг
- **$\log \sigma_i = \log(\sqrt{\mathcal{L}_i^{(0)}})$**: Якщо знаєте приблизні значення Loss на початку
- **$\log \sigma_i = -1$** (тобто $\sigma_i \approx 0.37$): Якщо хочете почати з більших ваг

### 7.3. Learning rate для $\log \sigma_i$

**Рекомендації:**

- Використовуйте **той самий learning rate**, що й для основної моделі
- Або **трохи менший** (наприклад, 0.1× основного LR) для стабільності

### 7.4. Моніторинг навчання

**Що відстежувати:**

1. **Ваги $w_i = \frac{1}{2\sigma_i^2}$:** Повинні стабілізуватися після кількох епох
2. **Співвідношення $\frac{\mathcal{L}_i}{\sigma_i^2}$:** Повинно прагнути до 1 (згідно з вправою 5.1)
3. **Градієнти по $\log \sigma_i$:** Повинні зменшуватися з часом

---

## 8. Висновки

### 8.1. Ключові ідеї

1. **Uncertainty-Weighted Loss** — це елегантне математичне рішення проблеми балансування компонентів Loss
2. **Теоретичне обґрунтування:** Виведено з байєсівської оптимізації та мінімізації вариації
3. **Автоматизація:** Не потребує ручного підбору ваг — модель навчається оптимальні значення
4. **Інтерпретація:** Параметри $\sigma_i$ мають чіткий фізичний зміст (невизначеність)

### 8.2. Практичні переваги

- ✅ **Економія часу:** Не потрібно тижнів експериментів з підбором ваг
- ✅ **Стабільність:** Автоматично адаптується до змін під час навчання
- ✅ **Універсальність:** Працює для різних задач та архітектур
- ✅ **Інваріантність:** Не залежить від масштабування компонентів

### 8.3. Наступні кроки

1. **Спробуйте Uncertainty-Weighted Loss** у своїх проектах
2. **Порівняйте** з фіксованими вагами та Gradient Norm Balancing
3. **Експериментуйте** з різними початковими значеннями $\log \sigma_i$
4. **Моніторьте** ваги та невизначеності під час навчання

---

## 9. Додаткові ресурси

### 9.1. Оригінальні статті

- **Kendall et al. (2018):** "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" — оригінальна стаття про Uncertainty-Weighted Loss
- **Kendall & Gal (2017):** "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" — теоретичний фундамент

### 9.2. Реалізації

- **PyTorch:** Реалізація в цьому семінарі
- **TensorFlow:** Аналогічна реалізація доступна в TensorFlow Probability

### 9.3. Пов'язані теми

- **Bayesian Neural Networks:** Теоретичний фундамент
- **Multi-Task Learning:** Застосування Uncertainty-Weighted Loss
- **Gradient Balancing:** Альтернативні підходи

---

**Кінець семінару**

