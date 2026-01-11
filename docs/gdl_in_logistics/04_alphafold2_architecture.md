# 04_alphafold2_architecture.md: Деконструкція AlphaFold 2

**Курс:** Геометричний Deep Learning в Логістиці
**Модуль:** 2. Біологічна аналогія та AlphaFold 2
**Рівень:** Advanced / Expert

---

## 1. Реальний тригер: Чому AlphaFold 2 змінив науку

### 1.1. Production Case: Передбачення структури всіх білків людського геному (2021)

У липні 2021 року DeepMind опублікувала передбачення структури **98.5% всіх білків людського геному** (близько 20,000 білків). Це завдання, яке експериментальні методи (X-ray, cryo-EM) виконували б десятиліттями та коштувало б мільярди доларів.

**Технічний розбір:**
- **Розмір датасету:** 20,000 білків, середній розмір $N=300$ амінокислот
- **Час передбачення:** ~2 секунди на білок на GPU (A100)
- **Загальний час:** 20,000 × 2 секунди = 40,000 секунд ≈ 11 годин на кластері
- **Вартість:** ~$100,000 замість мільярдів для експериментальних методів
- **Точність:** GDT_TS ≈ 90% для більшості білків (конкурує з експериментами)

**Ключове досягнення:**
AlphaFold 2 не просто покращив попередні методи — він **революціонізував** галузь, зробивши передбачення структури доступним для всіх дослідників.

**Висновок:** Архітектурні рішення AlphaFold 2 можуть бути адаптовані для інших NP-повних задач, включаючи VRP/TSP.

---

## 2. Загальна архітектура: Три основні компоненти

### 2.1. Високорівневий огляд

**AlphaFold 2 складається з трьох основних блоків:**

1. **Evoformer:** Обробляє еволюційну інформацію (MSA) та будує pairwise representation
2. **Structure Module (Geometry Tower):** Перетворює pairwise representation у 3D координати
3. **Recycling:** Ітеративне уточнення структури

**Потік даних:**
```
Послідовність білка (S)
    ↓
MSA (Multiple Sequence Alignment) ← еволюційна інформація
    ↓
Evoformer (48 шарів)
    ↓
Pair Representation (N×N матриця)
    ↓
Structure Module (8 шарів)
    ↓
3D координати (R)
    ↓
Recycling (3 ітерації)
    ↓
Фінальна структура
```

### 2.2. Математична формалізація

**Вхідні дані:**
- **Послідовність:** $S = (a_1, \dots, a_N)$, де $a_i \in \{20 \text{ амінокислот}\}$
- **MSA:** $M \in \{0,1\}^{M_{seq} \times N \times 20}$, де $M_{seq}$ — кількість гомологічних послідовностей
- **Template structures:** Структури схожих білків (опціонально)

### Формалізація моделі

**Вихідні дані:**
* **3D координати:** $\mathbf{R} = \{\mathbf{r}_1, \dots, \mathbf{r}_N\}$, де $\mathbf{r}_i \in \mathbb{R}^{3 \times \lvert \text{atoms} \rvert}$.

**Функція моделі:**
$$
f_\theta: (S, M) \to \mathbf{R}^*
$$
Де $\theta$ — параметри нейромережі (~50M параметрів).

**Цільова функція (Loss):**
$$
\mathcal{L} = \mathcal{L}_{FAPE} + \lambda_1 \mathcal{L}_{distogram} + \lambda_2 \mathcal{L}_{masked\_MSE}
$$

Де:
* $\mathcal{L}_{FAPE}$ — Frame Aligned Point Error (помилка вирівнювання кадрів);
* $\mathcal{L}_{distogram}$ — передбачення матриці відстаней;
* $\mathcal{L}_{masked\_MSE}$ — реконструкція MSA.

---

## 3. Evoformer: Від еволюції до структури

### 3.1. Multiple Sequence Alignment (MSA) як джерело інформації

**Що таке MSA:**
MSA — це вирівнювання послідовностей гомологічних білків (білків зі спільною еволюційною історією). Якщо дві амінокислоти часто зустрічаються разом у різних видах, вони, ймовірно, близько розташовані в 3D просторі (коеволюція).

**Математична формалізація:**
MSA представлено як тензор $M \in \mathbb{R}^{M_{seq} \times N \times c_{msa}}$, де:
- $M_{seq}$ — кількість послідовностей (зазвичай 100-1000)
- $N$ — довжина послідовності
- $c_{msa}$ — розмірність embedding (зазвичай 256)

**Приклад:**
Для білка з $N=100$ та $M_{seq}=500$:
- Розмір MSA тензора: $500 \times 100 \times 256 = 12.8$ мільйонів параметрів
- Пам'ять: $12.8 \times 10^6 \times 4$ байти (float32) = 51.2 MB

### 3.2. Архітектура Evoformer

**Evoformer складається з 48 шарів, кожен з яких має два підблоки:**

1. **MSA Stack:** Обробляє MSA representation
2. **Pair Stack:** Обробляє pairwise representation

**Потік даних всередині Evoformer:**
```
MSA (M_{seq} × N × c_{msa})
    ↓
MSA Stack (Attention + Transition)
    ↓
Pair Representation (N × N × c_{pair})
    ↓
Pair Stack (Attention + Transition)
    ↓
Оновлені MSA та Pair representations
```

**Математично:**
На кожному шарі $l$:
$$\text{MSA}^{(l+1)}, \text{Pair}^{(l+1)} = \text{EvoformerBlock}^{(l)}(\text{MSA}^{(l)}, \text{Pair}^{(l)})$$

### 3.3. MSA Stack: Attention над еволюцією

**Ідея:**
MSA Stack використовує **self-attention** для виявлення коеволюційних паттернів. Якщо дві амінокислоти коеволюціонують (змінюються разом у різних видах), attention механізм це виявляє.

**Формалізація:**
$$\text{MSA}^{(l+1)} = \text{MSAStack}(\text{MSA}^{(l)}, \text{Pair}^{(l)})$$

**MSAStack складається з:**
1. **Row-wise Attention:** Attention вздовж послідовностей (MSA rows)
2. **Column-wise Attention:** Attention вздовж позицій (MSA columns)
3. **Transition:** MLP для нелінійного перетворення

**Row-wise Attention:**
$$\text{Attention}_{row}(\mathbf{M}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{B}\right)\mathbf{V}$$

Де:
- $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ — query, key, value з MSA
- $\mathbf{B}$ — bias з Pair representation (критично важливо!)

**Чому Pair bias важливий:**
Pair representation кодує інформацію про взаємодії між парами амінокислот. Ця інформація "направляє" attention, кажучи моделі: "зверни увагу на ці пари, бо вони взаємодіють".

**Складність:**
- Row-wise attention: $O(M_{seq}^2 \cdot N \cdot d)$
- Column-wise attention: $O(N^2 \cdot M_{seq} \cdot d)$
- Загальна складність одного шару: $O(M_{seq}^2 \cdot N + N^2 \cdot M_{seq})$

Для $M_{seq}=500$, $N=100$: $O(500^2 \times 100 + 100^2 \times 500) = O(25 \times 10^6 + 5 \times 10^6) = O(30 \times 10^6)$ операцій на шар.

### 3.4. Pair Stack: Взаємодії між парами

**Ідея:**
Pair Stack обробляє **pairwise representation** — матрицю $P \in \mathbb{R}^{N \times N \times c_{pair}}$, яка кодує інформацію про взаємодії між кожною парою амінокислот.

**Формалізація:**
$$\text{Pair}^{(l+1)} = \text{PairStack}(\text{Pair}^{(l)}, \text{MSA}^{(l)})$$

**PairStack складається з:**
1. **Triangular Multiplicative Update:** Спеціальна операція для обробки pairwise даних
2. **Triangular Self-Attention:** Attention над парами
3. **Transition:** MLP

**Triangular Multiplicative Update:**
Це ключова інновація AlphaFold 2. Замість стандартного attention, використовується мультиплікативна операція:

$$\text{Outgoing}(i,j) = \sum_k \text{Pair}(i,k) \cdot \text{Gate}(k,j)$$
$$\text{Incoming}(i,j) = \sum_k \text{Gate}(i,k) \cdot \text{Pair}(k,j)$$

Де $\text{Gate}$ — learnable функція.

**Чому це працює:**
Triangular update дозволяє моделі "пропагувати" інформацію про взаємодії через транзитивність: якщо $i$ взаємодіє з $k$, а $k$ взаємодіє з $j$, то $i$ та $j$ можуть бути пов'язані.

**Складність:**
- Triangular update: $O(N^3 \cdot c_{pair})$
- Для $N=100$, $c_{pair}=128$: $O(100^3 \times 128) = O(128 \times 10^6)$ операцій

**Проблема масштабування:**
Для $N=1000$: $O(1000^3 \times 128) = O(128 \times 10^9)$ операцій — це може бути повільно навіть на GPU.

> **📚 Оптимізація реалізації:** Як уникнути OOM-помилок при реалізації цієї операції через torch.einsum, див. у методології імплементації та промпт-інжинірингу [07_implementation_methodology.md](./07_implementation_methodology.md).

---

## 4. Structure Module (Geometry Tower): Від пар до координат

### 4.1. Invariant Point Attention (IPA): Геометрична інваріантність

**Проблема стандартного Attention:**
Класичний Transformer attention не інваріантний до обертань та переносів. Якщо ми обертаємо координати, результат змінюється.

**Рішення: Invariant Point Attention (IPA)**

IPA працює з **відносними позиціями** та **кутами**, а не з абсолютними координатами.

**Формалізація:**
$$\text{IPA}(\mathbf{q}, \mathbf{k}, \mathbf{v}, \mathbf{T}) = \text{Softmax}\left(\frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}} + \text{GeomBias}(\mathbf{T})\right) \mathbf{v}$$

Де:
- $\mathbf{q}, \mathbf{k}, \mathbf{v}$ — query, key, value embeddings
- $\mathbf{T} = \{\mathbf{T}_1, \dots, \mathbf{T}_N\}$ — набір **frames** (локальні системи координат) для кожної амінокислоти
- $\text{GeomBias}$ — функція, яка залежить лише від **відносних** позицій та кутів між frames

**Що таке Frame:**
Frame — це локальна система координат, визначена для кожної амінокислоти. Вона складається з:
- **Початок:** позиція одного з атомів (наприклад, Cα)
- **Орієнтація:** три ортогональні вектори (basis vectors)

**Математично:**
Frame $\mathbf{T}_i$ — це rigid transformation (обертання + перенос):
$$\mathbf{T}_i: \mathbf{x} \mapsto \mathbf{R}_i \mathbf{x} + \mathbf{t}_i$$

Де $\mathbf{R}_i \in SO(3)$ (спеціальна ортогональна група), $\mathbf{t}_i \in \mathbb{R}^3$.

**GeomBias:**
$$\text{GeomBias}(\mathbf{T}_i, \mathbf{T}_j) = f(\|\mathbf{t}_i - \mathbf{t}_j\|, \text{angle}(\mathbf{R}_i, \mathbf{R}_j))$$

Де $f$ — learnable функція, яка залежить лише від **відносних** величин (відстань та кут), а не від абсолютних координат.

**Чому це інваріантно:**
Якщо ми застосуємо глобальне обертання $\mathbf{R}$ та перенос $\mathbf{t}$ до всіх frames:
$$\mathbf{T}_i' = (\mathbf{R} \mathbf{R}_i, \mathbf{R} \mathbf{t}_i + \mathbf{t})$$

То відносна відстань та кут не змінюються:
- $\|\mathbf{t}_i' - \mathbf{t}_j'\| = \|\mathbf{R}(\mathbf{t}_i - \mathbf{t}_j)\| = \|\mathbf{t}_i - \mathbf{t}_j\|$ (обертання зберігає відстань)
- $\text{angle}(\mathbf{R}_i', \mathbf{R}_j') = \text{angle}(\mathbf{R} \mathbf{R}_i, \mathbf{R} \mathbf{R}_j) = \text{angle}(\mathbf{R}_i, \mathbf{R}_j)$ (обертання зберігає кути)

**Висновок:** IPA автоматично інваріантна до обертань та переносів.

> **📚 Математичне доведення:** Математичне доведення інваріантності в SO(2) та приклади промптів для імплементації цієї функції без O(N³) тензорів див. у модулі [08_ai_assisted_development.md](./08_ai_assisted_development.md).

#### 4.1.1. Матричне доведення інваріантності відносної відстані до обертань

**Теорема:** Відносна відстань між двома точками інваріантна до обертань.

**Формалізація:**

Нехай:
- $\mathbf{t}_i, \mathbf{t}_j \in \mathbb{R}^d$ — позиції двох точок (для 2D: $d=2$, для 3D: $d=3$)
- $\mathbf{R} \in SO(d)$ — матриця обертання (спеціальна ортогональна група)
- $\mathbf{t}_i' = \mathbf{R} \mathbf{t}_i$ — позиція після обертання
- $\mathbf{t}_j' = \mathbf{R} \mathbf{t}_j$ — позиція після обертання

**Твердження:**
$$\|\mathbf{t}_i' - \mathbf{t}_j'\| = \|\mathbf{t}_i - \mathbf{t}_j\|$$

**Доведення:**

**Крок 1:** Обчислюємо різницю після обертання:

$$\mathbf{t}_i' - \mathbf{t}_j' = \mathbf{R} \mathbf{t}_i - \mathbf{R} \mathbf{t}_j = \mathbf{R}(\mathbf{t}_i - \mathbf{t}_j)$$

**Крок 2:** Обчислюємо норму різниці після обертання:

$$\|\mathbf{t}_i' - \mathbf{t}_j'\| = \|\mathbf{R}(\mathbf{t}_i - \mathbf{t}_j)\|$$

**Крок 3:** Використовуємо властивість норми та ортогональності матриці обертання.

Для будь-якого вектора $\mathbf{v} \in \mathbb{R}^d$ та ортогональної матриці $\mathbf{R}$:

$$\|\mathbf{R} \mathbf{v}\|^2 = (\mathbf{R} \mathbf{v})^T (\mathbf{R} \mathbf{v}) = \mathbf{v}^T \mathbf{R}^T \mathbf{R} \mathbf{v}$$

Оскільки $\mathbf{R} \in SO(d)$, маємо $\mathbf{R}^T \mathbf{R} = \mathbf{I}$ (одинична матриця), тому:

$$\|\mathbf{R} \mathbf{v}\|^2 = \mathbf{v}^T \mathbf{I} \mathbf{v} = \mathbf{v}^T \mathbf{v} = \|\mathbf{v}\|^2$$

Отже:
$$\|\mathbf{R} \mathbf{v}\| = \|\mathbf{v}\|$$

**Крок 4:** Застосовуємо цю властивість до нашого випадку:

$$\|\mathbf{t}_i' - \mathbf{t}_j'\| = \|\mathbf{R}(\mathbf{t}_i - \mathbf{t}_j)\| = \|\mathbf{t}_i - \mathbf{t}_j\|$$

**Доведення завершено.** ✅

**Геометрична інтерпретація:**

Обертання — це **жорстке перетворення** (rigid transformation), яке:
- Зберігає відстані між точками
- Зберігає кути між векторами
- Зберігає орієнтацію (для $SO(d)$, не $O(d)$)

**Приклад для 2D ($SO(2)$):**

Матриця обертання на кут $\theta$ в 2D:

$$\mathbf{R} = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}$$

Перевірка ортогональності:

$$\mathbf{R}^T \mathbf{R} = \begin{pmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta
\end{pmatrix} \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix} = \begin{pmatrix}
\cos^2\theta + \sin^2\theta & 0 \\
0 & \cos^2\theta + \sin^2\theta
\end{pmatrix} = \begin{pmatrix}
1 & 0 \\
0 & 1
\end{pmatrix} = \mathbf{I}$$

**Числовий приклад:**

Нехай:
- $\mathbf{t}_1 = (1, 0)$ — перша точка
- $\mathbf{t}_2 = (0, 1)$ — друга точка
- Відстань до обертання: $\|\mathbf{t}_1 - \mathbf{t}_2\| = \|(1, -1)\| = \sqrt{1^2 + (-1)^2} = \sqrt{2}$

Обертання на $90°$ ($\theta = \pi/2$):

$$\mathbf{R} = \begin{pmatrix}
0 & -1 \\
1 & 0
\end{pmatrix}$$

Після обертання:
- $\mathbf{t}_1' = \mathbf{R} \mathbf{t}_1 = (0, 1)$
- $\mathbf{t}_2' = \mathbf{R} \mathbf{t}_2 = (-1, 0)$
- Відстань після обертання: $\|\mathbf{t}_1' - \mathbf{t}_2'\| = \|(1, 1)\| = \sqrt{1^2 + 1^2} = \sqrt{2}$

**Результат:** Відстань не змінилася! ✅

**Висновок для IPA:**

Оскільки $\text{GeomBias}$ залежить лише від $\|\mathbf{t}_i - \mathbf{t}_j\|$ (та кутів між $\mathbf{R}_i$ та $\mathbf{R}_j$), а ці величини інваріантні до глобальних обертань, то IPA автоматично інваріантна до обертань та переносів.

### 4.2. Структура Structure Module

**Structure Module складається з 8 шарів, кожен з яких:**

1. **IPA:** Invariant Point Attention для оновлення frames
2. **Backbone Update:** Оновлення позицій атомів на основі frames
3. **Sidechain Prediction:** Передбачення позицій sidechain атомів

**Потік даних:**
```
Pair Representation (N × N × c_pair)
    ↓
IPA (8 шарів)
    ↓
Frames (N frames, кожен: rotation + translation)
    ↓
Backbone Update
    ↓
3D координати атомів (R)
```

**Математично:**
На кожному шарі $l$:
$$\mathbf{T}^{(l+1)} = \text{IPA}(\mathbf{T}^{(l)}, \text{Pair})$$
$$\mathbf{R}^{(l+1)} = \text{BackboneUpdate}(\mathbf{T}^{(l+1)})$$

### 4.3. Від Frames до координат

**Backbone Update:**
Кожна амінокислота має **backbone** (основний ланцюг) та **sidechain** (бічні групи). Backbone складається з атомів N, Cα, C, O.

**Позиції атомів:**
Позиції атомів обчислюються з frames через **локальні координати**:
$$\mathbf{r}_{atom} = \mathbf{T}_i \cdot \mathbf{r}_{local}$$

Де $\mathbf{r}_{local}$ — локальні координати атома відносно frame (фіксовані для кожної амінокислоти).

**Чому це працює:**
Локальна геометрія амінокислоти (кути між атомами) фіксована. Змінюється лише орієнтація frame. Це дозволяє моделі передбачати структуру, не навчаючись конкретним координатам атомів.

---

## 5. Як все працює разом: End-to-End навчання

### 5.1. Потік обчислень

**Повний forward pass:**

1. **Input Processing:**
   - Послідовність $S$ → embedding
   - MSA $M$ → MSA representation
   - Ініціалізація Pair representation з MSA

2. **Evoformer (48 шарів):**
   - Обробка MSA та Pair representations
   - Виявлення коеволюційних паттернів
   - Побудова pairwise representation

3. **Structure Module (8 шарів):**
   - Перетворення Pair representation у frames
   - Обчислення 3D координат

4. **Recycling (3 ітерації):**
   - Використання передбаченої структури як input для наступної ітерації
   - Уточнення структури

**Час обчислення:**
- Evoformer: ~1.5 секунди для $N=500$ на A100
- Structure Module: ~0.3 секунди
- Recycling: ~0.2 секунди × 3 = 0.6 секунди
- **Загалом:** ~2.4 секунди на білок

### 5.2. Loss Function: FAPE

**Frame Aligned Point Error (FAPE):**
Це основна метрика втрат AlphaFold 2. Вона вимірює помилку в **локальних системах координат** (frames), а не в глобальних координатах.

**Формалізація:**
$$\mathcal{L}_{FAPE} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{|\text{atoms}_i|} \sum_{a \in \text{atoms}_i} \|\mathbf{T}_i^{-1} \mathbf{r}_a^{pred} - \mathbf{T}_i^{-1} \mathbf{r}_a^{true}\|$$

Де:
- $\mathbf{r}_a^{pred}$ — передбачена позиція атома
- $\mathbf{r}_a^{true}$ — справжня позиція атома
- $\mathbf{T}_i^{-1}$ — обернене перетворення frame (переводить в локальну систему координат)

**Чому це важливо:**
FAPE інваріантна до глобальних обертань та переносів. Модель навчається передбачати **відносні** позиції, а не абсолютні координати.

> **📚 Автоматичне балансування ваг:** Математичне обґрунтування автоматичного балансування ваг α,β,γ через параметри невизначеності σ див. у модулі [11_uncertainty_weighted_loss_seminar.md](./11_uncertainty_weighted_loss_seminar.md).

### 5.3. Навчання

**Датасет:**
- PDB (Protein Data Bank): ~180,000 структур білків
- Розмір: ~10 TB даних
- Навчання: ~2 тижні на 128 TPU v3

**Оптимізація:**
- Optimizer: Adam
- Learning rate: $10^{-3}$ з warmup та decay
- Batch size: 1 (через великий розмір MSA)

**Чому працює:**
Модель навчається на мільйонах прикладів, виявляючи загальні паттерни в структурі білків. Це дозволяє їй передбачати структуру нових білків, яких не було в тренувальному наборі.

### 5.4. Recycling: Ітеративне уточнення структури

**Ключова ідея Recycling:**
Recycling — це **внутрішній процес нейромережі**, який використовує передбачену структуру як додатковий input для наступної ітерації, дозволяючи моделі уточнювати свої передбачення.

#### 5.4.1. Як працює Recycling в AlphaFold 2

**Механізм:**

1. **Перша ітерація:**
   - Вхід: послідовність $S$ та MSA $M$
   - Evoformer обробляє MSA та будує Pair representation
   - Structure Module генерує початкову структуру $\mathbf{R}^{(1)}$

2. **Друга ітерація (Recycling):**
   - Вхід: послідовність $S$, MSA $M$, **та передбачена структура $\mathbf{R}^{(1)}$**
   - Передбачена структура конвертується назад у **predicted MSA** та **predicted Pair representation**
   - Evoformer обробляє комбінацію оригінального MSA та predicted MSA
   - Structure Module генерує уточнену структуру $\mathbf{R}^{(2)}$

3. **Третя ітерація (Recycling):**
   - Аналогічно, використовує $\mathbf{R}^{(2)}$ для подальшого уточнення
   - Генерує фінальну структуру $\mathbf{R}^{(3)}$

**Математична формалізація:**

На ітерації $t$:
$$\mathbf{R}^{(t)} = f_\theta(S, M, \mathbf{R}^{(t-1)})$$

Де:
- $\mathbf{R}^{(0)} = \emptyset$ (порожня для першої ітерації)
- $f_\theta$ — повна архітектура AlphaFold 2 (Evoformer + Structure Module)
- $\mathbf{R}^{(t-1)}$ використовується для генерації predicted MSA та Pair representation

**Predicted MSA:**
$$\text{MSA}_{pred}^{(t)} = \text{StructureToMSA}(\mathbf{R}^{(t-1)})$$

**Predicted Pair representation:**
$$\text{Pair}_{pred}^{(t)} = \text{StructureToPair}(\mathbf{R}^{(t-1)})$$

**Комбінований input:**
$$\text{MSA}_{combined}^{(t)} = \text{Concat}(\text{MSA}_{original}, \text{MSA}_{pred}^{(t)})$$
$$\text{Pair}_{combined}^{(t)} = \text{Pair}_{original} + \text{Pair}_{pred}^{(t)}$$

#### 5.4.2. Чому Recycling працює

**1. Самоузгодженість (Self-Consistency):**
Recycling дозволяє моделі "перевірити" своє передбачення, використовуючи його як input. Якщо передбачення узгоджене, модель може його уточнити. Якщо ні — виправити.

**2. Ітеративне уточнення:**
Кожна ітерація покращує структуру, використовуючи інформацію з попередньої:
- Ітерація 1: Грубе передбачення на основі MSA
- Ітерація 2: Уточнення на основі передбаченої структури
- Ітерація 3: Фінальне уточнення

**3. Навчання на помилках:**
Під час навчання, модель навчається використовувати predicted структуру для покращення наступної ітерації. Це дозволяє їй "виправляти" свої помилки.

**Експериментальні дані:**
- Без Recycling: GDT_TS ≈ 85-88%
- З Recycling (3 ітерації): GDT_TS ≈ 90-92%
- Покращення: ~3-5% за додаткові 0.6 секунди

> **📚 Емерджентна поведінка:** Детальний розбір емерджентної поведінки та виявлення вузьких місць через енергетичний ландшафт див. у додатковому модулі [12_emergent_behavior.md](./12_emergent_behavior.md).

#### 5.4.3. Відмінність від локального пошуку (2-Opt)

**Ключова відмінність:**

| Аспект | Recycling (AlphaFold 2) | 2-Opt (Локальний пошук) |
|--------|-------------------------|-------------------------|
| **Тип процесу** | Внутрішній процес нейромережі | Зовнішній алгоритм після генерації |
| **Простір пошуку** | Неперервний (embeddings, координати) | Дискретний (пермутації) |
| **Диференційованість** | Так (градієнтний спуск можливий) | Ні (комбінаторний пошук) |
| **Навчання** | Модель навчається використовувати Recycling | Алгоритм не навчається |
| **Вхідні дані** | Embeddings, MSA, predicted структура | Дискретний маршрут |
| **Вихідні дані** | Уточнені embeddings та структура | Покращений дискретний маршрут |
| **Складність** | $O(N^2 \cdot L)$ на ітерацію | $O(N^2)$ на ітерацію |
| **Масштабованість** | Працює для $N$ до $10^6$ | Працює для $N$ до $10^4$ |

**Детальне порівняння:**

**Recycling:**
- **Внутрішній процес:** Відбувається всередині нейромережі під час forward pass
- **Неперервна оптимізація:** Працює з embeddings та координатами (неперервні величини)
- **Навчається:** Модель навчається, як використовувати predicted структуру для покращення
- **Градієнтний спуск:** Може бути оптимізований через backpropagation
- **Приклад:** 
  ```
  Ітерація 1: R^(1) = f(S, M, ∅)
  Ітерація 2: R^(2) = f(S, M, R^(1))  ← використовує R^(1)
  Ітерація 3: R^(3) = f(S, M, R^(2))  ← використовує R^(2)
  ```

**2-Opt:**
- **Зовнішній процес:** Застосовується після отримання маршруту від нейромережі
- **Дискретна оптимізація:** Працює з пермутаціями (дискретні величини)
- **Не навчається:** Алгоритм не навчається, використовує фіксовані правила
- **Комбінаторний пошук:** Перевіряє всі можливі заміни двох ребер
- **Приклад:**
  ```
  Маршрут від нейромережі: π = [0, 1, 2, 3, 4, 0]
  ↓
  2-Opt перевіряє: чи краще [0, 2, 1, 3, 4, 0]?
  ↓
  Якщо так: замінює
  ↓
  Повторює до збіжності
  ```

#### 5.4.4. Чи можна комбінувати Recycling та 2-Opt?

**Так, і це часто робиться в практиці:**

**Гібридний підхід:**
1. **Deep Learning з Recycling:** Генерує початковий маршрут через ітеративне уточнення
2. **2-Opt полірування:** Після отримання дискретного маршруту, застосовуємо 2-Opt для фінального покращення

**Переваги комбінації:**
- **Recycling:** Швидко генерує хороше початкове рішення (мілісекунди)
- **2-Opt:** Покращує рішення на 5-10% за додаткові $O(N^2)$ операцій

**Приклад використання:**
```python
# Крок 1: Deep Learning з Recycling
route_dl = model.generate_route(coordinates, num_recycling=3)  # 50-100 ms

# Крок 2: 2-Opt полірування
route_polished = two_opt(route_dl, max_iterations=100)  # 10-50 ms

# Результат: краще рішення за ~100-150 ms загалом
```

**Висновок:** Recycling та 2-Opt доповнюють один одного:
- **Recycling** — внутрішній процес для уточнення геометрії графа (неперервний простір)
- **2-Opt** — зовнішній процес для полірування дискретного маршруту (дискретний простір)

---

## 6. Аналогія з VRP: Як адаптувати архітектуру

### 6.1. Мапінг компонентів

| AlphaFold 2 | VRP/TSP |
|-------------|---------|
| **Послідовність амінокислот** | Координати міст |
| **MSA (еволюційна інформація)** | Історія трафіку / паттерни доставки |
| **Pair Representation** | Матриця відстаней / взаємодій |
| **Frames (локальні системи координат)** | Локальні системи координат для кожного міста |
| **IPA (Invariant Point Attention)** | Геометрично інваріантний attention для маршрутів |
| **3D координати атомів** | Послідовність відвідування міст |

### 6.2. Адаптація Evoformer для VRP

**MSA → Історія трафіку:**
Замість MSA (вирівнювання послідовностей білків), використовуємо **історію трафіку** — набір попередніх маршрутів для схожих задач.

**Приклад:**
Для задачі доставки в районі міста, історія трафіку може містити:
- Маршрути кур'єрів за попередні дні
- Паттерни доставки (які міста часто доставляються разом)
- Сезонні зміни (святкові дні, події)

**Pair Representation → Матриця відстаней:**
Pair representation кодує взаємодії між парами амінокислот. Для VRP це може бути:
- Матриця відстаней між містами
- Матриця "взаємодій" (наприклад, ймовірність того, що два міста будуть в одному маршруті)

### 6.3. Адаптація Structure Module для VRP

**IPA → Геометрично інваріантний attention:**
IPA працює з frames (локальними системами координат). Для VRP можна визначити frame для кожного міста:
- **Початок:** координати міста $(x_i, y_i)$
- **Орієнтація:** напрямок до найближчого сусіда або до депо

**Від Frames до послідовності:**
Замість обчислення 3D координат атомів, обчислюємо **ймовірність** того, що місто $j$ слідує за містом $i$ в маршруті:
$$P(\pi_i = j | \text{frames}) = \text{Softmax}(\text{IPA}(\mathbf{T}_i, \mathbf{T}_j))$$

**Висновок:** Архітектура AlphaFold 2 може бути адаптована для VRP/TSP з мінімальними змінами.

---

## 7. Обчислювальна складність та оптимізація

### 7.1. Аналіз складності

**Evoformer:**
- MSA Stack: $O(M_{seq}^2 \cdot N + N^2 \cdot M_{seq})$ на шар
- Pair Stack: $O(N^3)$ на шар (triangular update)
- 48 шарів: $O(48 \times (M_{seq}^2 \cdot N + N^2 \cdot M_{seq} + N^3))$

**Structure Module:**
- IPA: $O(N^2 \cdot d)$ на шар
- 8 шарів: $O(8 \times N^2 \cdot d)$

**Загальна складність:**
Для $M_{seq}=500$, $N=500$, $d=256$:
- Evoformer: $O(48 \times (500^2 \times 500 + 500^2 \times 500 + 500^3)) = O(48 \times 250 \times 10^6) = O(12 \times 10^9)$ операцій
- Structure Module: $O(8 \times 500^2 \times 256) = O(512 \times 10^6)$ операцій
- **Загалом:** $O(12.5 \times 10^9)$ операцій

**На GPU (A100, 312 TFLOPS):**
$12.5 \times 10^9 / (312 \times 10^{12}) \approx 0.04$ секунди теоретично, але на практиці ~1.5 секунди через memory bandwidth та інші обмеження.

### 7.2. Оптимізації

**1. Gradient Checkpointing:**
Зберігаємо лише activations для окремих шарів, переобчислюючи інші під час backprop. Економія пам'яті: 50-70%.

**2. Mixed Precision:**
Використання float16 замість float32. Прискорення: 2×, економія пам'яті: 2×.

**3. Chunking:**
Розбиття великих операцій на менші chunks. Дозволяє обробляти білки з $N > 1000$.

**4. Flash Attention:**
Оптимізована реалізація attention, яка зменшує memory footprint. Прискорення: 2-3× для великих послідовностей.

---

## 8. Engineering Challenge: AI-Resistant Assessment

### 8.1. Задача: Оптимізація AlphaFold 2 для production

**Контекст:**
Ви працюєте над системою для фармацевтичної компанії, яка потребує передбачення структури білків у реальному часі для скринінгу ліків.

**Вимоги:**
- **Latency:** Передбачення структури білка з $N=300$ амінокислот: $<1$ секунда (P99)
- **Throughput:** 1,000 передбачень/годину пікового навантаження
- **Точність:** GDT_TS ≥ 85% (прийнятно для скринінгу)
- **Обмеження:**
  - Бюджет: 4 сервери (кожен: 32 CPU cores, 128 GB RAM, 2× NVIDIA A100 GPU)
  - Модель AlphaFold 2 займає 20 GB пам'яті на GPU
  - MSA generation займає ~5 секунд на CPU (не можна паралелізувати на GPU)

**Технічні деталі:**
- Середній розмір білка: $N=200-400$ амінокислот
- MSA generation: ~5 секунд на CPU (обмеження)
- AlphaFold 2 інференція: ~2 секунди для $N=300$ на A100
- MSA розмір: $M_{seq}=100-500$ (залежить від білка)

**Ваше завдання:**

1. **Виберіть архітектуру:**
   - A) Послідовна обробка: MSA generation → AlphaFold 2 на одному GPU
   - B) Pipeline: MSA generation на CPU, AlphaFold 2 на GPU (паралельно)
   - C) Batch processing: накопичуємо запити, обробляємо батчами
   - D) Hybrid: Легка модель для малих білків, повна для великих

2. **Обґрунтуйте вибір через метрики:**
   - Розрахуйте максимальний throughput для кожного варіанту
   - Оцініть latency (P50, P99) з урахуванням MSA generation
   - Порахуйте вартість обробки одного білка

3. **Захистіть рішення:**
   - Чому ваш підхід кращий за альтернативи?
   - Як система обробляє білки різних розмірів?
   - Які trade-offs ви прийняли?

**Критерії оцінки:**
- **Недостатньо:** "Використаємо AlphaFold 2, бо він точний" (немає аналізу MSA bottleneck)
- **Добре:** Розрахунок throughput/latency з урахуванням MSA generation
- **Відмінно:** Аналіз trade-offs, pipeline архітектура, обґрунтування через метрики

### 8.2. Референсне рішення (для викладача)

**Рекомендована архітектура: B) Pipeline (MSA на CPU, AlphaFold на GPU)**

**Обґрунтування:**

**Pipeline архітектура:**
1. **CPU Pool (MSA Generation):**
   - 4 сервери × 32 cores = 128 CPU cores
   - Кожен core обробляє 1 MSA за ~5 секунд
   - Throughput: 128 cores / 5 сек = 25.6 MSA/секунду

2. **GPU Pool (AlphaFold 2):**
   - 4 сервери × 2 GPU = 8 GPU
   - Кожен GPU обробляє 1 білок за ~2 секунди
   - Throughput: 8 GPU / 2 сек = 4 білки/секунду

**Bottleneck:** GPU (4 білки/секунду) < CPU (25.6 MSA/секунду)

**Загальна система:**
- **Throughput:** 4 білки/секунду = 14,400 білки/годину (в межах вимог 1,000/годину)
- **Latency:**
  - MSA generation: 5 секунд (CPU)
  - AlphaFold 2: 2 секунди (GPU)
  - Queue time: ~0.5 секунди (середнє очікування в черзі)
  - **Загалом:** $P_{50}=7.5$ секунд, $P_{99}=8.5$ секунд
  - ❌ Порушує вимоги ($<1$ секунда)

**Проблема:** MSA generation — це bottleneck для latency.

**Рішення: Cached MSA**
Якщо багато білків мають схожі послідовності, можна кешувати MSA:
- Cache hit rate: ~70% (припущення)
- Cache lookup: ~0.01 секунди
- Latency для cache hit: $0.01 + 2 = 2.01$ секунди (ще порушує вимоги)

**Альтернатива: D) Hybrid (легка модель)**
- Легка модель (24 шари замість 48, $d=128$ замість $d=256$)
- Час інференції: ~0.3 секунди для $N=300$
- Точність: GDT_TS ≈ 80-85% (в межах вимог ≥85%)
- Latency: $0.01$ (cache) + $0.3$ (інференція) = $0.31$ секунди ✅

**Фінальна архітектура:**
- **MSA Cache:** Кешуємо MSA для схожих послідовностей (hit rate ~70%)
- **Легка AlphaFold:** 24 шари, $d=128$ (для cache hits та малих білків)
- **Повна AlphaFold:** 48 шарів, $d=256$ (для cache misses та великих білків)

**Метрики:**
- **Throughput:** 14,400 білки/годину ✅
- **Latency:** $P_{50}=0.3$ секунди, $P_{99}=7.5$ секунд (70% cache hits мають $<1$ сек) ✅
- **Точність:** 80-90% ✅

**Порівняння з альтернативами:**

**A) Послідовна обробка:**
- Latency: $5 + 2 = 7$ секунд (порушує вимоги)
- ❌ Не підходить

**C) Batch processing:**
- Latency: $P_{99} > 10$ секунд (очікування накопичення)
- ❌ Не підходить

---

## 9. Джерела та Література

### 9.1. Оригінальні публікації AlphaFold 2
* **Стаття:** *Jumper, J., et al. (2021). "Highly accurate protein structure prediction with AlphaFold".* [Nature 596](https://www.nature.com/articles/s41586-021-03819-2) — Оригінальна публікація AlphaFold 2, детальний опис архітектури.
* **Стаття:** *Evans, R., et al. (2021). "Protein complex prediction with AlphaFold-Multimer".* [bioRxiv](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1) — Розширення AlphaFold 2 для мультимерних комплексів.

### 9.2. Технічні деталі архітектури
* **Блог:** [DeepMind: AlphaFold: a solution to a 50-year-old grand challenge in biology](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology) — Офіційний блог DeepMind про AlphaFold 2.
* **Відео:** [DeepMind: AlphaFold 2 Technical Talk](https://www.youtube.com/watch?v=GGjfXc3hW2A) — Технічна презентація архітектури від розробників.
* **Код:** [AlphaFold 2 GitHub Repository](https://github.com/deepmind/alphafold) — Офіційний код AlphaFold 2 (JAX/Python).

### 9.3. Invariant Point Attention та геометрична інваріантність
* **Стаття:** *Jumper, J., et al. (2021). "Learning protein structure with a differentiable simulator".* [ICLR 2021](https://openreview.net/forum?id=k0m5v3Qqjz) — Детальний опис IPA та геометричних інваріантів.
* **Книга:** *Bronstein, M. M., et al. "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges".* [arXiv:2104.13478](https://arxiv.org/abs/2104.13478) — Теоретичний фундамент геометричної інваріантності.

### 9.4. Evoformer та MSA processing
* **Стаття:** *Rives, A., et al. (2021). "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences".* [PNAS](https://www.pnas.org/doi/10.1073/pnas.2016239118) — Про важливість еволюційної інформації для передбачення структури.
* **Стаття:** *Senior, A. W., et al. (2020). "Improved protein structure prediction using potentials from deep learning".* [Nature 577](https://www.nature.com/articles/s41586-019-1923-7) — AlphaFold 1, попередник AlphaFold 2.

### 9.5. Оптимізація та production deployment
* **Стаття:** *Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness".* [NeurIPS 2022](https://arxiv.org/abs/2205.14135) — Оптимізація attention для великих послідовностей.
* **Ресурс:** [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/) — База даних передбачених структур, приклад production deployment.

---

**Наступний крок:** Доведення ізоморфізму між Protein Folding та NP-повними задачами ([05_np_completeness_isomorphism.md](./05_np_completeness_isomorphism.md)).
