# 07_implementation_methodology.md: Методологія імплементації (Deep Dive)

**Курс:** Геометричний Deep Learning в Логістиці
**Модуль:** 3. Синтез та Адаптація
**Рівень:** Advanced / Expert

---

## 1. Реальний тригер: Чому наївна імплементація Evoformer впала в production

### 1.1. Production Case: OOM (Out of Memory) при обробці 100 міст (2023)

У 2023 році команда розробки системи маршрутизації зіткнулася з катастрофою: модель, яка працювала на тестових даних ($N=20$ міст), падала з `CUDA out of memory` при спробі обробити реальні задачі ($N=100$ міст).

**Технічний розбір:**
- **Архітектура:** Адаптований Evoformer (24 шари History Stack, 24 шари Distance Stack)
- **Проблема:** Distance Stack використовував Triangular Multiplicative Update з наївною реалізацією
- **Розмір задачі:** $N=100$ міст, $M_{hist}=200$ історичних маршрутів
- **Пам'ять:** Очікувана: ~20 GB, реальна: >80 GB (перевищення в 4 рази)

**Корінь проблеми:**
Triangular Multiplicative Update створював проміжні тензори розміру $N \times N \times N \times c_{pair}$ замість $N \times N \times c_{pair}$. Для $N=100$, $c_{pair}=128$:
- Очікувана пам'ять: $100^2 \times 128 \times 4$ байт = 5.1 MB на шар
- Реальна пам'ять: $100^3 \times 128 \times 4$ байт = 512 MB на шар
- 24 шари: $512 \times 24 = 12.3$ GB лише для проміжних тензорів

**Висновок:** Наївна імплементація математично еквівалентна, але архітектурно неприйнятна. Потрібна оптимізація memory layout.

---

## 2. Triangular Multiplicative Update: Математика та оптимізація пам'яті

> **📚 Промпт-інжиніринг:** Приклади промптів для генерації цього коду без створення тензорів розмірності O(N³) див. у розділі [08_ai_assisted_development.md](./08_ai_assisted_development.md).

### 2.1. Формальне визначення

**Triangular Multiplicative Update (TMU)** — це операція, яка оновлює pairwise representation через транзитивність взаємодій.

**Математична формалізація:**
Нехай $P \in \mathbb{R}^{N \times N \times c}$ — pairwise representation (матриця відстаней/взаємодій).

**Outgoing Update:**
$$P_{ij}^{out} = \sum_{k=1}^{N} P_{ik} \odot G_{kj}$$

**Incoming Update:**
$$P_{ij}^{in} = \sum_{k=1}^{N} G_{ik} \odot P_{kj}$$

Де:
- $G \in \mathbb{R}^{N \times N \times c}$ — gate tensor (learnable)
- $\odot$ — element-wise multiplication (Hadamard product)

**Фінальне оновлення:**
$$P^{(l+1)} = \text{LayerNorm}(P^{(l)} + P^{out} + P^{in})$$

**Інтерпретація:**
Якщо місто $i$ близьке до $k$ ($P_{ik}$ велике), а $k$ близьке до $j$ ($G_{kj}$ велике), то $i$ та $j$ можуть бути пов'язані через транзитивність. Це дозволяє моделі виявляти **непрямі** зв'язки між містами.

### 2.2. Наївна імплементація (неправильно)

```python
def triangular_update_naive(P, G):
    """
    P: [N, N, c] - pairwise representation
    G: [N, N, c] - gate tensor
    """
    N, c = P.shape[0], P.shape[2]
    P_out = torch.zeros_like(P)
    P_in = torch.zeros_like(P)
    
    # Outgoing: P[i,j] = sum_k P[i,k] * G[k,j]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                P_out[i, j] += P[i, k] * G[k, j]  # [c] * [c] = [c]
    
    # Incoming: P[i,j] = sum_k G[i,k] * P[k,j]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                P_in[i, j] += G[i, k] * P[k, j]
    
    return P_out, P_in
```

**Аналіз складності:**
- **Час:** $O(N^3 \cdot c)$ — три вкладені цикли
- **Пам'ять:** $O(N^2 \cdot c)$ для $P$, $P_{out}$, $P_{in}$ — прийнятно
- **Проблема:** На практиці PyTorch створює проміжні тензори розміру $[N, N, N, c]$ для broadcasting, що призводить до OOM

**Чому це відбувається:**
При операції `P[i, k] * G[k, j]`, PyTorch намагається broadcast тензори до розміру $[N, N, N, c]$ перед множенням. Це виникає через неявний broadcasting у циклах.

### 2.3. Оптимізована імплементація (правильно)

**Ключова ідея:** Використовуємо матричне множення замість циклів.

**Математична еквівалентність:**
Outgoing update можна переписати як:
$$P_{ij}^{out} = \sum_{k=1}^{N} P_{ik} \odot G_{kj} = \sum_{k=1}^{N} \sum_{d=1}^{c} P_{ikd} \cdot G_{kjd}$$

Це еквівалентно матричному множенню для кожного каналу $d$:
$$P_{:,:,d}^{out} = P_{:,:,d} \cdot G_{:,:,d}^T$$

**Оптимізований код:**
```python
def triangular_update_optimized(P, G):
    """
    P: [N, N, c] - pairwise representation
    G: [N, N, c] - gate tensor
    """
    # Outgoing: P[i,j] = sum_k P[i,k] * G[k,j]
    # Для кожного каналу: P[:,:,d] @ G[:,:,d].T
    P_out = torch.einsum('ikd,kjd->ijd', P, G)
    
    # Incoming: P[i,j] = sum_k G[i,k] * P[k,j]
    # Для кожного каналу: G[:,:,d] @ P[:,:,d]
    P_in = torch.einsum('ikd,kjd->ijd', G, P)
    
    return P_out, P_in
```

**Аналіз складності:**
- **Час:** $O(N^3 \cdot c)$ — така ж складність, але з меншою константою (оптимізовані BLAS операції)
- **Пам'ять:** $O(N^2 \cdot c)$ — **без проміжних тензорів** розміру $N^3$
- **Прискорення:** 10-50× на GPU завдяки оптимізації cuBLAS

**Чому `einsum` працює:**
`torch.einsum` оптимізує порядок операцій та уникає створення проміжних тензорів. Для `'ikd,kjd->ijd'` він виконує множення безпосередньо, не створюючи тензор розміру $[N, N, N, c]$.

### 2.4. Memory Layout: Чому порядок каналів має значення

**Проблема:** Навіть з `einsum`, порядок каналів у тензорі впливає на швидкість.

**Варіант A: Channels Last (неправильно для цієї операції)**
```python
P = torch.randn(N, N, c)  # [N, N, c] - channels last
```
**Проблема:** При множенні $P_{:,:,d} \cdot G_{:,:,d}^T$, ми читаємо пам'ять не послідовно (strided access), що сповільнює операцію.

**Варіант B: Channels First (правильно)**
```python
P = torch.randn(c, N, N)  # [c, N, N] - channels first
P_out = torch.einsum('dik,dkj->dij', P, G)
```
**Перевага:** Кожна матриця $P_{d,:,:}$ лежить в пам'яті послідовно, що дозволяє використовувати vectorized операції.

**Вимірювання:**
Для $N=100$, $c=128$ на A100 GPU:
- Channels Last: ~2.5 ms на шар
- Channels First: ~0.8 ms на шар
- **Прискорення: 3×**

**Висновок:** Memory layout критично важливий для продуктивності. Channels First для pairwise operations, Channels Last для node features.

---

## 3. Loss Functions для VRP: Математична формалізація та обчислювальна складність

### 3.1. Tour Length Loss: Від послідовності до вартості

**Проблема:** Як обчислити вартість маршруту $\pi = (\pi_1, \dots, \pi_N)$ ефективно?

**Наївний підхід:**
```python
def tour_length_naive(pi, distance_matrix):
    """
    pi: [N] - послідовність відвідування
    distance_matrix: [N, N] - матриця відстаней
    """
    cost = 0.0
    for i in range(len(pi) - 1):
        cost += distance_matrix[pi[i], pi[i+1]]
    cost += distance_matrix[pi[-1], pi[0]]  # Повернення в депо
    return cost
```

**Складність:** $O(N)$ — оптимально за часом, але не векторізується.

**Векторизований підхід:**
```python
def tour_length_vectorized(pi, distance_matrix):
    """
    pi: [batch_size, N] - батч послідовностей
    distance_matrix: [batch_size, N, N] - батч матриць відстаней
    """
    # Витягуємо ребра маршруту
    from_indices = pi  # [batch_size, N]
    to_indices = torch.roll(pi, shifts=-1, dims=1)  # [batch_size, N]
    
    # Gather відстані для кожного ребра
    batch_indices = torch.arange(pi.shape[0])[:, None].expand_as(pi)
    costs = distance_matrix[batch_indices, from_indices, to_indices]  # [batch_size, N]
    
    return costs.sum(dim=1)  # [batch_size]
```

**Аналіз:**
- **Час:** $O(N)$ — така ж складність, але паралельно для всього батчу
- **Пам'ять:** $O(\text{batch_size} \times N)$ для проміжних тензорів
- **Прискорення:** На GPU з batch_size=64: ~64× швидше за послідовну обробку

**Математична формалізація:**
$$\mathcal{L}_{length}(\pi, D) = \sum_{k=1}^{N-1} D_{\pi_k, \pi_{k+1}} + D_{\pi_N, \pi_1}$$

Де $D \in \mathbb{R}^{N \times N}$ — матриця відстаней.

### 3.2. Permutation Loss: Negative Log-Likelihood

**Проблема:** Модель генерує ймовірності $P(\pi_k = j | \pi_{<k})$, але нам потрібен loss для навчання.

**Autoregressive Generation:**
На кожному кроці $k$, модель обчислює:
$$P(\pi_k = j | \pi_{<k}, \mathbf{H}, \mathbf{D}) = \text{Softmax}(\text{Attention}(\mathbf{h}_j, \{\mathbf{h}_i : i \in \text{unvisited}\}))$$

Де $\mathbf{H}$ — node embeddings після Evoformer, $\mathbf{D}$ — pairwise representation.

**Negative Log-Likelihood Loss:**
$$\mathcal{L}_{perm} = -\sum_{k=1}^{N} \log P(\pi_k^{true} | \pi_{<k}^{true}, \mathbf{H}, \mathbf{D})$$

**Інтерпретація:**
Це стандартний cross-entropy loss для послідовності. Модель навчається максимізувати ймовірність правильного вибору на кожному кроці.

**Обчислювальна складність:**
- Для кожного кроку $k$: $O(N \cdot d)$ для attention (де $d$ — розмірність embeddings)
- Для всієї послідовності: $O(N^2 \cdot d)$
- Для батчу: $O(\text{batch_size} \times N^2 \cdot d)$

**Оптимізація через Teacher Forcing:**
Під час навчання використовуємо правильну послідовність $\pi^{true}$ для обчислення attention (teacher forcing), що дозволяє паралелізувати обчислення для всіх $k$ одночасно:

```python
def permutation_loss_parallel(logits, pi_true, mask):
    """
    logits: [batch_size, N, N] - логіти для кожного кроку
    pi_true: [batch_size, N] - правильна послідовність
    mask: [batch_size, N, N] - маска для невідвіданих міст
    """
    # Gather логіти для правильних виборів
    batch_indices = torch.arange(logits.shape[0])[:, None]
    step_indices = torch.arange(logits.shape[1])[None, :]
    selected_logits = logits[batch_indices, step_indices, pi_true]  # [batch_size, N]
    
    # Обчислюємо log-softmax з маскою
    masked_logits = logits.masked_fill(~mask, float('-inf'))
    log_probs = F.log_softmax(masked_logits, dim=-1)
    
    # Negative log-likelihood
    nll = -log_probs[batch_indices, step_indices, pi_true].sum(dim=1)
    
    return nll.mean()
```

**Складність:** $O(\text{batch_size} \times N^2)$ — паралельно для всіх кроків.

### 3.3. Constraint Loss: Soft Penalties для обмежень

**Проблема:** Як інтегрувати обмеження (часові вікна, вантажопідйомність) у loss function?

**Математична формалізація:**
Для маршруту $\pi$, час прибуття в місто $\pi_k$:
$$\text{arrival}_k = \sum_{i=1}^{k-1} t_{\pi_i, \pi_{i+1}} + \text{service}_{\pi_i}$$

Де $t_{ij}$ — час проїзду від $i$ до $j$, $\text{service}_i$ — час обслуговування в місті $i$.

**Constraint Loss для часових вікон:**
$$\mathcal{L}_{time} = \lambda_{time} \sum_{k=1}^{N} \left[\max(0, e_{\pi_k} - \text{arrival}_k) + \max(0, \text{arrival}_k - l_{\pi_k})\right]$$

Де $[e_i, l_i]$ — часове вікно для міста $i$.

**Constraint Loss для вантажопідйомності:**
$$\mathcal{L}_{capacity} = \lambda_{cap} \sum_{r=1}^{K} \max(0, \sum_{i \in \text{route}_r} q_i - Q)$$

Де $K$ — кількість транспортних засобів, $q_i$ — попит клієнта $i$, $Q$ — вантажопідйомність.

**Векторизована імплементація:**
```python
def constraint_loss(pi, arrival_times, time_windows, demands, capacity):
    """
    pi: [batch_size, N] - послідовності
    arrival_times: [batch_size, N] - час прибуття для кожного міста
    time_windows: [batch_size, N, 2] - [earliest, latest] для кожного міста
    demands: [batch_size, N] - попит для кожного міста
    capacity: scalar - вантажопідйомність
    """
    # Часові вікна
    early_violations = F.relu(time_windows[:, :, 0] - arrival_times)  # [batch_size, N]
    late_violations = F.relu(arrival_times - time_windows[:, :, 1])  # [batch_size, N]
    time_loss = (early_violations + late_violations).sum(dim=1).mean()
    
    # Вантажопідйомність (спрощено для одного транспортного засобу)
    route_demands = demands.gather(1, pi)  # [batch_size, N]
    cumulative_demands = route_demands.cumsum(dim=1)  # [batch_size, N]
    capacity_violations = F.relu(cumulative_demands - capacity)  # [batch_size, N]
    capacity_loss = capacity_violations.sum(dim=1).mean()
    
    return time_loss, capacity_loss
```

**Складність:** $O(\text{batch_size} \times N)$ — лінійна.

### 3.4. Комбінований Loss: Ваги та балансування

**Загальний Loss:**
$$\mathcal{L}_{total} = \alpha \mathcal{L}_{length} + \beta \mathcal{L}_{perm} + \gamma \mathcal{L}_{time} + \delta \mathcal{L}_{capacity}$$

**Вибір ваг:**
- $\alpha = 1.0$ — основний loss (вартість маршруту)
- $\beta = 0.1$ — допоміжний loss (правильність послідовності)
- $\gamma = 1.0$ — важливість обмежень (часові вікна)
- $\delta = 1.0$ — важливість обмежень (вантажопідйомність)

**Проблема масштабування:**
Різні компоненти loss мають різні масштаби:
- $\mathcal{L}_{length}$: може бути $10^3 - 10^5$ (метри/секунди)
- $\mathcal{L}_{perm}$: зазвичай $1 - 10$ (negative log-probability)
- $\mathcal{L}_{time}$: може бути $10^2 - 10^4$ (секунди порушення)

**Рішення: Нормалізація:**
$$\mathcal{L}_{total} = \alpha \frac{\mathcal{L}_{length}}{\text{baseline}_{length}} + \beta \mathcal{L}_{perm} + \gamma \frac{\mathcal{L}_{time}}{\text{baseline}_{time}} + \delta \frac{\mathcal{L}_{capacity}}{\text{baseline}_{capacity}}$$

Де `baseline` — середнє значення на валідаційному сеті.

**Адаптивні ваги (Gradient Balancing):**
Замість фіксованих ваг, використовуємо адаптивні:
$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\lambda \cdot \frac{\partial \mathcal{L}_i}{\partial \theta})$$

Це дозволяє автоматично балансувати внесок кожного компонента.

### 3.5. Стабільність навчання: Adaptive Gradient Balancing

**Проблема домінування компонентів:**

При використанні комбінованого Loss з фіксованими вагами $\alpha, \beta, \gamma, \delta$, один компонент може домінувати над іншими, що призводить до:

1. **Домінування Length Loss:** Модель ігнорує обмеження (часові вікна, вантажопідйомність) і мінімізує тільки довжину маршруту
2. **Домінування Constraint Loss:** Модель стає "занадто обережною", створюючи довгі маршрути, щоб уникнути порушень
3. **Нестабільне навчання:** Градієнти різних компонентів мають різні масштаби, що призводить до коливань

**Приклад проблеми:**

Нехай на кроці $t$:
- $\mathcal{L}_{length} = 5000$ (великий)
- $\mathcal{L}_{time} = 50$ (малий)
- $\alpha = 1.0$, $\gamma = 1.0$

Тоді внесок до загального Loss:
- $\alpha \mathcal{L}_{length} = 5000$ (домінує)
- $\gamma \mathcal{L}_{time} = 50$ (ігнорується)

Модель буде оптимізувати тільки довжину, ігноруючи часові вікна.

#### 3.5.1. Математична формалізація проблеми

**Градієнт загального Loss:**

$$\frac{\partial \mathcal{L}_{total}}{\partial \theta} = \alpha \frac{\partial \mathcal{L}_{length}}{\partial \theta} + \beta \frac{\partial \mathcal{L}_{perm}}{\partial \theta} + \gamma \frac{\partial \mathcal{L}_{time}}{\partial \theta} + \delta \frac{\partial \mathcal{L}_{capacity}}{\partial \theta}$$

**Проблема:** Якщо $\|\alpha \frac{\partial \mathcal{L}_{length}}{\partial \theta}\| \gg \|\gamma \frac{\partial \mathcal{L}_{time}}{\partial \theta}\|$, то градієнт домінується першим компонентом.

**Мета Adaptive Gradient Balancing:**

Балансувати **норми градієнтів** різних компонентів, щоб кожен компонент мав рівний вплив на оновлення параметрів.

#### 3.5.2. Gradient Balancing через норми градієнтів

**Ідея:** Нормалізувати ваги так, щоб норми градієнтів були приблизно рівні.

**Алгоритм:**

На кожному кроці $t$:

1. Обчислюємо градієнти для кожного компонента:
   $$\mathbf{g}_{length}^{(t)} = \frac{\partial \mathcal{L}_{length}}{\partial \theta}, \quad \mathbf{g}_{perm}^{(t)} = \frac{\partial \mathcal{L}_{perm}}{\partial \theta}, \quad \mathbf{g}_{time}^{(t)} = \frac{\partial \mathcal{L}_{time}}{\partial \theta}, \quad \mathbf{g}_{capacity}^{(t)} = \frac{\partial \mathcal{L}_{capacity}}{\partial \theta}$$

2. Обчислюємо норми градієнтів:
   $$n_{length}^{(t)} = \|\mathbf{g}_{length}^{(t)}\|, \quad n_{perm}^{(t)} = \|\mathbf{g}_{perm}^{(t)}\|, \quad n_{time}^{(t)} = \|\mathbf{g}_{time}^{(t)}\|, \quad n_{capacity}^{(t)} = \|\mathbf{g}_{capacity}^{(t)}\|$$

3. Обчислюємо середню норму (або максимальну):
   $$\bar{n}^{(t)} = \frac{n_{length}^{(t)} + n_{perm}^{(t)} + n_{time}^{(t)} + n_{capacity}^{(t)}}{4}$$

4. Адаптуємо ваги:
   $$\alpha^{(t+1)} = \alpha^{(t)} \cdot \frac{\bar{n}^{(t)}}{n_{length}^{(t)} + \epsilon}$$
   $$\beta^{(t+1)} = \beta^{(t)} \cdot \frac{\bar{n}^{(t)}}{n_{perm}^{(t)} + \epsilon}$$
   $$\gamma^{(t+1)} = \gamma^{(t)} \cdot \frac{\bar{n}^{(t)}}{n_{time}^{(t)} + \epsilon}$$
   $$\delta^{(t+1)} = \delta^{(t)} \cdot \frac{\bar{n}^{(t)}}{n_{capacity}^{(t)} + \epsilon}$$

Де $\epsilon = 10^{-8}$ — мала константа для стабільності.

**Реалізація:**

```python
class AdaptiveGradientBalancer:
    def __init__(self, initial_weights, momentum=0.9):
        """
        initial_weights: dict з початковими вагами {'length': 1.0, 'perm': 0.1, ...}
        momentum: коефіцієнт для експоненційного згладжування
        """
        self.weights = initial_weights.copy()
        self.momentum = momentum
        self.ema_norms = {key: 1.0 for key in initial_weights.keys()}
    
    def update_weights(self, gradients):
        """
        gradients: dict з градієнтами для кожного компонента
        {'length': grad_tensor, 'perm': grad_tensor, ...}
        """
        # Обчислюємо норми градієнтів
        norms = {}
        for key, grad in gradients.items():
            if grad is not None:
                norms[key] = grad.norm().item()
            else:
                norms[key] = self.ema_norms[key]  # Використовуємо попереднє значення
        
        # Експоненційне згладжування норм (для стабільності)
        for key in norms:
            self.ema_norms[key] = (
                self.momentum * self.ema_norms[key] + 
                (1 - self.momentum) * norms[key]
            )
        
        # Обчислюємо середню норму
        mean_norm = sum(self.ema_norms.values()) / len(self.ema_norms)
        
        # Адаптуємо ваги
        epsilon = 1e-8
        for key in self.weights:
            if self.ema_norms[key] > epsilon:
                self.weights[key] *= mean_norm / self.ema_norms[key]
        
        return self.weights
```

#### 3.5.3. Gradient Balancing через відносні зміни

**Альтернативний підхід:** Балансувати не норми градієнтів, а **відносні зміни** компонентів Loss.

**Алгоритм:**

1. Обчислюємо відносні зміни Loss компонентів:
   $$r_i^{(t)} = \frac{|\mathcal{L}_i^{(t)} - \mathcal{L}_i^{(t-1)}|}{\mathcal{L}_i^{(t-1)} + \epsilon}$$

2. Якщо один компонент зменшується швидше за інші, збільшуємо його вагу:
   $$w_i^{(t+1)} = w_i^{(t)} \cdot (1 + \lambda \cdot (r_{mean}^{(t)} - r_i^{(t)}))$$

Де $r_{mean}^{(t)}$ — середнє відносних змін, $\lambda$ — швидкість адаптації (зазвичай 0.01-0.1).

**Переваги:**

- Автоматично збільшує вагу компонентів, які "відстають"
- Стабільніший за чистий gradient balancing

#### 3.5.4. Uncertainty-Weighted Loss Balancing

> **📚 Математичне обґрунтування:** Математичне обґрунтування автоматичного балансування ваг через параметри невизначеності σ див. у семінарі [11_uncertainty_weighted_loss_seminar.md](./11_uncertainty_weighted_loss_seminar.md).

**Ідея:** Використовувати **невизначеність** (uncertainty) моделі для автоматичного балансування.

**Математична формалізація:**

Замість фіксованих ваг, вводимо **learnable параметри невизначеності** $\sigma_i$:

$$\mathcal{L}_{total} = \sum_{i} \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i$$

Де:
- $\frac{1}{\sigma_i^2}$ — автоматична вага для компонента $i$
- $\log \sigma_i$ — regularization термін (штрафує за надто великі $\sigma_i$)

**Інтерпретація:**

- Якщо $\sigma_i$ великий → $\frac{1}{\sigma_i^2}$ малий → компонент має менший вплив
- Якщо $\sigma_i$ малий → $\frac{1}{\sigma_i^2}$ великий → компонент має більший вплив

Модель автоматично навчається балансувати компоненти через оптимізацію $\sigma_i$.

**Реалізація:**

```python
class UncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_components=4):
        super().__init__()
        # Learnable параметри невизначеності
        self.log_sigmas = nn.Parameter(torch.zeros(num_components))
    
    def forward(self, losses):
        """
        losses: dict з компонентами Loss
        {'length': tensor, 'perm': tensor, 'time': tensor, 'capacity': tensor}
        """
        total_loss = 0.0
        sigmas = torch.exp(self.log_sigmas)  # Забезпечуємо позитивність
        
        for i, (key, loss) in enumerate(losses.items()):
            weight = 1.0 / (2 * sigmas[i]**2 + 1e-8)
            total_loss += weight * loss + self.log_sigmas[i]
        
        return total_loss
```

#### 3.5.5. Gradient Clipping для стабільності

**Додаткова техніка:** Обмежувати градієнти компонентів, щоб уникнути екстремальних значень.

**Алгоритм:**

```python
def balanced_loss_with_clipping(losses, weights, max_grad_norm=1.0):
    """
    Обчислює загальний Loss з gradient clipping
    """
    total_loss = sum(w * loss for w, loss in zip(weights.values(), losses.values()))
    
    # Під час backward, застосовуємо gradient clipping
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    return total_loss
```

#### 3.5.6. Порівняння методів балансування

**Таблиця порівняння:**

| Метод | Складність | Стабільність | Автоматичність | Застосування |
|-------|------------|--------------|----------------|-------------|
| **Фіксовані ваги** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ | Прості задачі |
| **Gradient Norm Balancing** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Загальне застосування |
| **Relative Change Balancing** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Коли компоненти мають різні швидкості збіжності |
| **Uncertainty Weighting** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Коли потрібна максимальна автоматизація |

**Рекомендації:**

1. **Почати з фіксованих ваг + нормалізації** (розділ 3.4)
2. **Якщо навчання нестабільне:** Додати Gradient Norm Balancing
3. **Якщо компоненти збігаються з різною швидкістю:** Використати Relative Change Balancing
4. **Для максимальної автоматизації:** Uncertainty-Weighted Loss

#### 3.5.7. Практичний приклад: Навчання з Adaptive Balancing

**Сценарій:** VRP з часовими вікнами, де Length Loss домінує над Time Loss.

**Без балансування:**
```python
# Фіксовані ваги
alpha, gamma = 1.0, 1.0

# На кроці t:
L_length = 5000  # Великий
L_time = 50      # Малий

# Загальний Loss
L_total = 1.0 * 5000 + 1.0 * 50 = 5050
# Length домінує (99% внеску)
```

**З Gradient Norm Balancing:**
```python
balancer = AdaptiveGradientBalancer({
    'length': 1.0,
    'time': 1.0
})

# На кроці t:
g_length = compute_gradient(L_length)  # ||g_length|| = 100
g_time = compute_gradient(L_time)      # ||g_time|| = 1

# Адаптація ваг
mean_norm = (100 + 1) / 2 = 50.5
alpha_new = 1.0 * 50.5 / 100 = 0.505
gamma_new = 1.0 * 50.5 / 1 = 50.5

# Тепер ваги балансовані
L_total = 0.505 * 5000 + 50.5 * 50 = 2525 + 2525 = 5050
# Обидва компоненти мають рівний вплив на градієнт
```

**Результат:** Модель навчається враховувати обидва компоненти рівномірно, не ігноруючи часові вікна.

#### 3.5.8. Моніторинг балансування під час навчання

**Ключові метрики для відстеження:**

1. **Відношення норм градієнтів:**
   $$r_{ij}^{(t)} = \frac{\|\mathbf{g}_i^{(t)}\|}{\|\mathbf{g}_j^{(t)}\| + \epsilon}$$

   Ідеальне значення: $r_{ij} \approx 1$ (градієнти мають подібні норми)

2. **Відносний внесок до загального Loss:**
   $$p_i^{(t)} = \frac{w_i \mathcal{L}_i}{\sum_j w_j \mathcal{L}_j}$$

   Ідеальне значення: $p_i \approx \frac{1}{N}$ для $N$ компонентів (рівномірний розподіл)

3. **Коефіцієнт варіації ваг:**
   $$CV^{(t)} = \frac{\sigma(w^{(t)})}{\mu(w^{(t)})}$$

   Де $\sigma$ — стандартне відхилення, $\mu$ — середнє значення ваг.

   Інтерпретація:
   - $CV < 0.5$: Ваги добре балансовані
   - $CV > 1.0$: Один компонент домінує

**Візуалізація під час навчання:**

```python
def log_balancing_metrics(epoch, losses, weights, gradients, writer):
    """
    Логує метрики балансування для TensorBoard/Weights & Biases
    """
    # 1. Норми градієнтів
    grad_norms = {key: grad.norm().item() for key, grad in gradients.items()}
    
    # 2. Відносні внески до Loss
    total_weighted_loss = sum(w * losses[key] for key, w in weights.items())
    loss_contributions = {
        key: (weights[key] * losses[key]) / total_weighted_loss 
        for key in losses.keys()
    }
    
    # 3. Коефіцієнт варіації ваг
    weight_values = torch.tensor(list(weights.values()))
    weight_mean = weight_values.mean().item()
    weight_std = weight_values.std().item()
    cv = weight_std / (weight_mean + 1e-8)
    
    # Логування
    for key in losses.keys():
        writer.add_scalar(f'grad_norm/{key}', grad_norms[key], epoch)
        writer.add_scalar(f'loss_contribution/{key}', loss_contributions[key], epoch)
        writer.add_scalar(f'weight/{key}', weights[key], epoch)
    
    writer.add_scalar('balancing/cv_weights', cv, epoch)
    
    # Попередження, якщо балансування не працює
    if cv > 1.0:
        print(f"⚠️ Warning: Unbalanced weights detected (CV={cv:.2f}) at epoch {epoch}")
```

**Індикатори проблем:**

1. **Один компонент домінує:**
   - $p_i > 0.8$ для деякого $i$
   - $r_{ij} > 10$ або $r_{ij} < 0.1$ для деяких $i, j$

2. **Ваги нестабільні:**
   - Різкі стрибки ваг між епохами
   - $CV$ коливається без збіжності

3. **Компоненти не збігаються:**
   - Один Loss зменшується, інший залишається постійним або зростає

**Рекомендації з налаштування:**

- **Якщо $CV > 1.0$:** Збільшити швидкість адаптації (більший $\lambda$)
- **Якщо ваги коливаються:** Зменшити швидкість адаптації або додати більше momentum
- **Якщо один компонент не збігається:** Перевірити, чи він правильно обчислюється, можливо додати більшу початкову вагу

---

## 4. Покроковий гайд адаптації Evoformer

### 4.1. Крок 1: Підготовка даних (Data Preprocessing)

**Вхідні дані:**
- Координати міст: $\mathbf{X} \in \mathbb{R}^{N \times 2}$
- Історія трафіку: $\mathbf{H} \in \mathbb{R}^{M_{hist} \times N \times c_{hist}}$ (опціонально)
- Матриця відстаней: $\mathbf{D} \in \mathbb{R}^{N \times N}$ (або обчислюється з координат)

**Крок 1.1: Нормалізація координат**
```python
def normalize_coordinates(X):
    """
    X: [N, 2] - координати міст
    """
    # Центруємо навколо центру мас
    center = X.mean(dim=0)  # [2]
    X_centered = X - center
    
    # Масштабуємо до одиничного квадрата
    scale = X_centered.abs().max()
    X_normalized = X_centered / (scale + 1e-8)
    
    return X_normalized, center, scale
```

**Чому це важливо:**
- Модель навчається на нормалізованих даних
- Градієнти стабільніші (немає великих значень)
- Інваріантність до масштабу (автоматично)

**Крок 1.2: Обчислення матриці відстаней**
```python
def compute_distance_matrix(X, metric='euclidean'):
    """
    X: [N, 2] - координати міст
    """
    if metric == 'euclidean':
        # ||x_i - x_j||_2
        D = torch.cdist(X, X, p=2)  # [N, N]
    elif metric == 'manhattan':
        D = torch.cdist(X, X, p=1)  # [N, N]
    else:
        # Використовуємо OSRM або інший routing engine
        D = compute_road_distance(X)
    
    return D
```

**Складність:** $O(N^2)$ для евклідової відстані, $O(N^2 \times \text{OSRM latency})$ для дорожньої відстані.

**Крок 1.3: Підготовка історії трафіку**
```python
def prepare_traffic_history(historical_routes, current_cities):
    """
    historical_routes: List[List[int]] - список історичних маршрутів
    current_cities: [N] - індекси поточних міст
    """
    # Фільтруємо історію, залишаючи лише релевантні маршрути
    relevant_routes = [r for r in historical_routes if set(r).issubset(set(current_cities))]
    
    # Створюємо embeddings для кожного маршруту
    M_hist = len(relevant_routes)
    H = torch.zeros(M_hist, len(current_cities), c_hist)
    
    for m, route in enumerate(relevant_routes):
        for i, city in enumerate(route):
            city_idx = current_cities.index(city)
            H[m, city_idx, :] = compute_route_embedding(route, i)
    
    return H
```

### 4.2. Крок 2: Node Embeddings (Initial Features)

**Перетворення координат у embeddings:**
```python
class NodeEmbedding(nn.Module):
    def __init__(self, coord_dim=2, feature_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, X, additional_features=None):
        """
        X: [batch_size, N, 2] - координати
        additional_features: [batch_size, N, d_feat] - додаткові features (попит, часові вікна)
        """
        h = self.mlp(X)  # [batch_size, N, feature_dim]
        
        if additional_features is not None:
            h = h + self.feature_proj(additional_features)
        
        return h
```

**Пам'ять:** $O(\text{batch_size} \times N \times d)$ для embeddings.

### 4.3. Крок 3: History Stack (Адаптований MSA Stack)

**Архітектура:**
```python
class HistoryStack(nn.Module):
    def __init__(self, num_layers=12, c_msa=64, c_pair=128):
        super().__init__()
        self.layers = nn.ModuleList([
            HistoryLayer(c_msa, c_pair) for _ in range(num_layers)
        ])
    
    def forward(self, H, D):
        """
        H: [batch_size, M_hist, N, c_msa] - історія трафіку
        D: [batch_size, N, N, c_pair] - pairwise representation
        """
        for layer in self.layers:
            H, D = layer(H, D)
        return H, D
```

**HistoryLayer (адаптований MSA Layer):**
```python
class HistoryLayer(nn.Module):
    def __init__(self, c_msa, c_pair):
        super().__init__()
        # Row-wise attention (вздовж історичних маршрутів)
        self.row_attention = MultiHeadAttention(c_msa, num_heads=8)
        
        # Column-wise attention (вздовж міст)
        self.col_attention = MultiHeadAttention(c_msa, num_heads=8)
        
        # Transition (MLP)
        self.transition = nn.Sequential(
            nn.Linear(c_msa, c_msa * 4),
            nn.ReLU(),
            nn.Linear(c_msa * 4, c_msa)
        )
        
        # Distance bias
        self.distance_bias = nn.Linear(c_pair, 1)
    
    def forward(self, H, D):
        """
        H: [batch_size, M_hist, N, c_msa]
        D: [batch_size, N, N, c_pair]
        """
        # Row-wise attention з distance bias
        D_bias = self.distance_bias(D).squeeze(-1)  # [batch_size, N, N]
        H = self.row_attention(H, bias=D_bias)
        
        # Column-wise attention
        H = H.transpose(1, 2)  # [batch_size, N, M_hist, c_msa]
        H = self.col_attention(H)
        H = H.transpose(1, 2)  # [batch_size, M_hist, N, c_msa]
        
        # Transition
        H = H + self.transition(H)
        
        return H, D
```

**Складність:** $O(\text{batch_size} \times M_{hist}^2 \times N + \text{batch_size} \times N^2 \times M_{hist})$ на шар.

### 4.4. Крок 4: Distance Stack (Адаптований Pair Stack)

**Архітектура:**
```python
class DistanceStack(nn.Module):
    def __init__(self, num_layers=12, c_pair=128):
        super().__init__()
        self.layers = nn.ModuleList([
            DistanceLayer(c_pair) for _ in range(num_layers)
        ])
    
    def forward(self, D, H):
        """
        D: [batch_size, N, N, c_pair] - pairwise representation
        H: [batch_size, M_hist, N, c_msa] - історія трафіку
        """
        for layer in self.layers:
            D = layer(D, H)
        return D
```

**DistanceLayer з Triangular Multiplicative Update:**
```python
class DistanceLayer(nn.Module):
    def __init__(self, c_pair):
        super().__init__()
        self.triangular_update = TriangularMultiplicativeUpdate(c_pair)
        self.self_attention = MultiHeadAttention(c_pair, num_heads=8)
        self.transition = nn.Sequential(
            nn.Linear(c_pair, c_pair * 4),
            nn.ReLU(),
            nn.Linear(c_pair * 4, c_pair)
        )
    
    def forward(self, D, H):
        """
        D: [batch_size, N, N, c_pair]
        H: [batch_size, M_hist, N, c_msa] - для bias
        """
        # Triangular Multiplicative Update
        D = D + self.triangular_update(D)
        
        # Self-attention (опціонально)
        D_flat = D.view(-1, D.shape[2], D.shape[3])  # [batch_size * N, N, c_pair]
        D_flat = self.self_attention(D_flat)
        D = D_flat.view(D.shape)
        
        # Transition
        D = D + self.transition(D)
        
        return D
```

**TriangularMultiplicativeUpdate (оптимізована версія):**
```python
class TriangularMultiplicativeUpdate(nn.Module):
    def __init__(self, c_pair):
        super().__init__()
        self.gate = nn.Linear(c_pair, c_pair)
    
    def forward(self, P):
        """
        P: [batch_size, N, N, c_pair]
        """
        # Обчислюємо gate
        G = torch.sigmoid(self.gate(P))  # [batch_size, N, N, c_pair]
        
        # Outgoing: P[i,j] = sum_k P[i,k] * G[k,j]
        P_out = torch.einsum('bikd,bkjd->bijd', P, G)
        
        # Incoming: P[i,j] = sum_k G[i,k] * P[k,j]
        P_in = torch.einsum('bikd,bkjd->bijd', G, P)
        
        # LayerNorm
        P_updated = F.layer_norm(P + P_out + P_in, (P.shape[-1],))
        
        return P_updated
```

**Пам'ять:** $O(\text{batch_size} \times N^2 \times c_{pair})$ — без проміжних тензорів $N^3$.

### 4.5. Крок 5: Route Module (Адаптований Structure Module)

**Генерація послідовності відвідування:**
```python
class RouteModule(nn.Module):
    def __init__(self, num_layers=8, c_pair=128, c_node=128):
        super().__init__()
        self.layers = nn.ModuleList([
            RouteLayer(c_pair, c_node) for _ in range(num_layers)
        ])
        self.route_head = nn.Linear(c_node, 1)  # Для генерації ймовірностей
    
    def forward(self, D, node_embeddings):
        """
        D: [batch_size, N, N, c_pair] - pairwise representation
        node_embeddings: [batch_size, N, c_node] - node features
        """
        # Обчислюємо frames (локальні системи координат)
        frames = self.compute_frames(node_embeddings, D)
        
        # Geodesic Attention
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, frames, D)
        
        # Генерація ймовірностей
        logits = self.route_head(node_embeddings)  # [batch_size, N, 1]
        
        return logits.squeeze(-1)  # [batch_size, N]
```

**Autoregressive Generation:**
```python
def generate_route_autoregressive(model, node_embeddings, D, start=0):
    """
    Генерація маршруту покроково
    """
    batch_size, N = node_embeddings.shape[:2]
    pi = torch.zeros(batch_size, N, dtype=torch.long)
    visited = torch.zeros(batch_size, N, dtype=torch.bool)
    
    current = torch.full((batch_size,), start, dtype=torch.long)
    pi[:, 0] = current
    visited.scatter_(1, current.unsqueeze(1), True)
    
    for k in range(1, N):
        # Обчислюємо ймовірності для невідвіданих міст
        logits = model(node_embeddings, D, current, visited)
        logits = logits.masked_fill(visited, float('-inf'))
        
        # Sampling (або argmax для детерміністичного)
        probs = F.softmax(logits, dim=-1)
        next_city = torch.multinomial(probs, 1).squeeze(-1)
        
        pi[:, k] = next_city
        visited.scatter_(1, next_city.unsqueeze(1), True)
        current = next_city
    
    return pi
```

**Складність:** $O(N^2 \cdot d)$ для генерації одного маршруту.

---

## 5. Engineering Challenge: AI-Resistant Assessment

### 5.1. Задача: Оптимізація пам'яті для Evoformer на обмеженому GPU

**Контекст:**
Ви маєте обмежений бюджет: 1× NVIDIA A100 GPU (40 GB пам'яті). Потрібно обробити VRP задачі з $N=200$ міст, $M_{hist}=300$ історичних маршрутів.

**Вимоги:**
- Batch size: мінімум 8 instances одночасно
- Latency: $<200$ ms для одного батчу
- Точність: не погіршити порівняно з повною моделлю

**Технічні параметри:**
- History Stack: 24 шари, $c_{msa}=128$
- Distance Stack: 24 шари, $c_{pair}=256$
- Route Module: 8 шарів, $c_{node}=128$

**Ваше завдання:**

1. **Розрахуйте потребу в пам'яті:**
   - Для кожного компонента (History Stack, Distance Stack, Route Module)
   - Для проміжних тензорів (activations, gradients)
   - Для оптимізатора (Adam state)

2. **Запропонуйте оптимізації:**
   - Gradient Checkpointing: де саме застосувати?
   - Mixed Precision Training: які операції можна в FP16?
   - Dynamic Batching: як адаптувати batch size до розміру задачі?
   - Model Parallelism: чи можна розділити модель між GPU?

3. **Обґрунтуйте trade-offs:**
   - Як кожна оптимізація впливає на latency?
   - Як впливає на точність?
   - Яка комбінація дає найкращий баланс?

**Критерії оцінки:**
- **Недостатньо:** "Використаємо gradient checkpointing" (немає розрахунків)
- **Добре:** Детальний розрахунок пам'яті з конкретними числами
- **Відмінно:** Аналіз trade-offs, обґрунтування через метрики, врахування latency

### 5.2. Референсне рішення (для викладача)

**Розрахунок пам'яті (базова модель):**

**1. History Stack:**
- Input: $8 \times 300 \times 200 \times 128 \times 4$ байт = 245.8 MB
- Output (24 шари): $24 \times 245.8$ MB = 5.9 GB
- **Проблема:** Перевищення пам'яті вже на цьому етапі

**2. Distance Stack:**
- Input: $8 \times 200 \times 200 \times 256 \times 4$ байт = 327.7 MB
- Output (24 шари): $24 \times 327.7$ MB = 7.9 GB

**3. Route Module:**
- Input: $8 \times 200 \times 128 \times 4$ байт = 819.2 KB
- Output (8 шарів): $8 \times 819.2$ KB = 6.6 MB

**4. Gradients:**
- Параметри моделі: ~50M параметрів × 4 байт = 200 MB
- Gradients: 200 MB
- Adam state: 400 MB (2× для momentums)

**5. Activations (проміжні тензори):**
- Attention matrices: $8 \times 24 \times 200^2 \times 4$ байт = 307.2 MB
- MLP activations: ~500 MB

**Загалом:** ~15 GB для одного батчу (без оптимізацій) — **не поміщається в 40 GB з batch_size=8**

**Оптимізації:**

**1. Gradient Checkpointing:**
Зберігаємо activations лише для кожного 4-го шару, решту перераховуємо під час backward:
- Економія: ~60% пам'яті на activations
- Overhead: +30% часу на forward/backward
- **Результат:** ~9 GB для батчу

**2. Mixed Precision (FP16):**
Використовуємо FP16 для activations та ваг (крім критичних операцій):
- Економія: 50% пам'яті
- Overhead: мінімальний (Tensor Cores на A100)
- **Результат:** ~4.5 GB для батчу

**3. Dynamic Batching:**
Для $N=200$ використовуємо batch_size=4, для $N=100$ — batch_size=8:
- Економія: 50% пам'яті для великих задач
- Overhead: менший throughput, але прийнятний

**4. Оптимізація Triangular Update:**
Використовуємо channels-first layout та `einsum`:
- Економія: ~200 MB на шар Distance Stack
- Overhead: мінімальний (швидше завдяки оптимізації)

**Фінальна конфігурація:**
- Gradient Checkpointing: кожні 4 шари
- Mixed Precision: FP16 для activations, FP32 для критичних операцій
- Dynamic Batching: batch_size=4 для $N=200$
- **Результат:** ~6 GB для батчу, latency: ~180 ms ✅

**Trade-offs:**
- ✅ Пам'ять: поміщається в 40 GB
- ✅ Latency: 180 ms < 200 ms
- ⚠️ Точність: -0.5% через mixed precision (прийнятно)
- ⚠️ Throughput: -25% через gradient checkpointing (прийнятно)

---

## 6. Джерела та Література

### 6.1. Triangular Multiplicative Update та оптимізація пам'яті
* **Стаття:** *Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold".* [Nature 2021](https://www.nature.com/articles/s41586-021-03819-2) — Оригінальний опис Triangular Update в AlphaFold 2.
* **Стаття:** *Chen, T., et al. (2016). "Training Deep Nets with Sublinear Memory Cost".* [arXiv:1604.06174](https://arxiv.org/abs/1604.06174) — Gradient Checkpointing для економії пам'яті.
* **Документація:** [PyTorch Mixed Precision Training](https://pytorch.org/docs/stable/amp.html) — Офіційний гайд по FP16 training.

### 6.2. Loss Functions для комбінаторної оптимізації
* **Стаття:** *Kool, W., et al. (2019). "Attention, Learn to Solve Routing Problems!".* [ICLR 2019](https://arxiv.org/abs/1803.08475) — Negative log-likelihood loss для TSP.
* **Стаття:** *Nazari, M., et al. (2018). "Reinforcement Learning for Solving the Vehicle Routing Problem".* [NeurIPS 2018](https://arxiv.org/abs/1802.04240) — REINFORCE loss для VRP.
* **Стаття:** *Chen, X., & Tian, Y. (2019). "Learning to Perform Local Rewriting for Combinatorial Optimization".* [NeurIPS 2019](https://arxiv.org/abs/1904.12314) — Local search через gradient descent.

### 6.3. Оптимізація пам'яті та продуктивності
* **Стаття:** *Kirisame, M., et al. (2020). "Dynamic Tensor Rematerialization".* [ICLR 2020](https://arxiv.org/abs/2006.09616) — Автоматична оптимізація пам'яті.
* **Ресурс:** [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/) — Оптимізації для GPU.
* **Стаття:** *Wang, M., et al. (2019). "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks".* [arXiv:1909.01315](https://arxiv.org/abs/1909.01315) — Оптимізації для GNN.

---

**Наступний крок:** Практикум з AI-асистованої розробки ([08_ai_assisted_development.md](./08_ai_assisted_development.md)).

