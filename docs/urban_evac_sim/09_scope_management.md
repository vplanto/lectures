---
title: "09 Scope Management"
type: project
module: Module 9
prerequisites: module 8
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# 09. Scope Management: Coursework vs Diploma

**Scope:** Project Roadmap & Grading Criteria.
**Goal:** Define the MVP for the Term Paper to avoid "scope creep".

## 1. Філософія розподілу

Ми розглядаємо цей проект як двоетапний R&D процес.
1.  **Курсова робота (The Engine):** Побудова фізичного світу. Головне — щоб симуляція працювала коректно з точки зору правил (паливо, смерть, затор). Швидкодія та "розум" системи вторинні.
2.  **Дипломна робота (The Brain):** Оптимізація та управління. Головне — масштабування до 10k агентів (Numba) та реалізація режимів "System Optimum".

---

## 2. Курсова робота (Minimum Viable Chaos)

**Девіз:** "It runs, and agents die."
**Вимога до масштабу:** $N \approx 500$ агентів. Python-цикли дозволені (якщо FPS > 1).

### Функціональний мінімум:
1.  **Topology:**
    * Завантаження шматка карти через `OSMnx`.
    * Перетворення у граф `NetworkX`.
    * *S simplification:* Можна без складного `Smart Contraction`.
2.  **Agent Physics (Critical):**
    * Рух з точки А в Б.
    * **Fuel Logic:** Витрата палива в русі та простої.
    * **Death Mechanic:** Якщо паливо = 0, агент стає статичною перешкодою. **Це головна фіча для захисту.**
3.  **Routing:**
    * Static $A^*$ (розраховується один раз на старті).
    * Ніякого динамічного перерахунку (Re-routing).
4.  **Analytics:**
    * Одна фінальна цифра: $S_{rate}$ (відсоток тих, хто вижив).
    * Простий графік: Кількість активних агентів vs Час.

**Що НЕ потрібно для курсової:**
* Складні перехрестя (можна вважати, що всі перехрестя — рівнозначні).
* Векторизація та Numba (чистий Python OK для 500 машин).
* Regime C (централізоване управління).

---

## 3. Дипломна робота (High-Load Simulation)

**Девіз:** "It scales, and we control the chaos."
**Вимога до масштабу:** $N \ge 10,000$ агентів. Real-time (Numba/JIT).

### Розширений функціонал:
1.  **Optimization (Performance):**
    * Перехід на NumPy векторизацію (Zero-loop physics).
    * Впровадження Spatial Hashing для пошуку сусідів.
2.  **Logic Upgrade:**
    * Реалізація **Regime B** (Hybrid) та **Regime C** (System Optimum).
    * Логіка світлофорів та пріоритетів проїзду.
    * Динамічний Re-routing (агенти шукають об'їзд заторів).
3.  **Deep Analytics:**
    * Порівняння сценаріїв (Price of Anarchy).
    * Heatmaps (карти щільності).
    * Аналіз "Вакуумного ефекту" та "Трафік-тріажу".

---

## 4. Матриця критеріїв (Checklist)

| Feature | Курсова (MVP) | Диплом (Final Product) |
| :--- | :--- | :--- |
| **Map Source** | OSM Small Area (1 district) | Full City / Large Area |
| **Population** | ~500 Agents | 10,000+ Agents |
| **Tech Stack** | Python / NetworkX | NumPy / Numba |
| **Fuel Physics** | **Required** (Basic) | Advanced (Idle vs Active) |
| **Routing** | Static Shortest Path | Dynamic / Load Balancing |
| **Intersections**| Simple "Pass through" | "Don't Block the Box" logic |
| **Output** | Console Logs + Static Plot | Interactive Dashboard |

---

## 5. Ризики та Pivot

Якщо студент не встигає реалізувати "System Optimum" навіть на диплом:
* **План Б:** Зосередитися на глибокому аналізі "Хаосу".
* Зробити якісну візуалізацію того, як саме "вмирає" місто.
* Побудувати графік залежності "Смертності" від "Кількості палива на старті".
* Це теж валідна наукова робота (дослідження вразливостей), навіть без створення алгоритму порятунку.
