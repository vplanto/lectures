---
title: "Практикум: Генерація керованого хаосу та синтетичних аномалій"
layout: default
author: Віталій Платонов
---

# Практикум: Генерація керованого хаосу та синтетичних аномалій

## 1. Chaos Engineering як верифікатор Спостережуваність (Observability)

### Факт

 Ви не можете бути впевнені у надійності своїх алертів та моделей BERT, доки не перевірите їх у "бойових" умовах. Традиційне тестування перевіряє функціональність; **Chaos Engineering** перевіряє спостережуваність (Спостережуваність (Observability)) та здатність системи до самовідновлення (Resilience). Мета цього лабу — створити контрольоване середовище для імітації каскадних збоїв та S-подібного розпаду продуктивності.

---

## 2. Python Chaos Agent: Імплементація деструктивних сценаріїв

Для тестування ми розробимо атомарний агент, здатний імітувати три типи збоїв: вибух кардинальності, логістичний розпад затримки та OOM-ситуації.

```python
import time
import random
import threading
import numpy as np
import requests # Припускаємо наявність експортера для Prometheus

class ChaosMachine:
    def __init__(self, service_name="order-processor"):
        self.service_name = service_name
        self.is_running = True
        self.metrics = {"latency": 0.05, "error_count": 0}

    def simulate_cpu_spike(self, duration=30):
        """Імітація High CPU через обчислювально складну задачу."""
        print(f"[*] Starting CPU Spike on {self.service_name}")
        end_time = time.time() + duration
        while time.time() < end_time:
            _ = [np.sqrt(x) for x in range(10000)]
        print("[!] CPU Spike finished")

    def simulate_memory_leak(self, increment_mb=50, interval=2):
        """Імітація OOM-ситуації (Linear growth)."""
        print(f"[*] Starting Memory Leak simulation")
        leak_buffer = []
        try:
            while self.is_running:
                leak_buffer.append(' ' * (increment_mb * 1024 * 1024))
                print(f"[-] Leaked {len(leak_buffer) * increment_mb} MB")
                time.sleep(interval)
        except MemoryError:
            print("[CRITICAL] Process OOMKilled simulation triggered")

    def simulate_logistic_latency_decay(self, K=2.0, r=0.3, t0=15):
        """
        Імітація S-подібного зростання затримки (Growth Theory).
        Моделює каскадний збій, що стабілізується.
        """
        print(f"[*] Starting Logistic Затримка (Latency) Decay (K={K}, r={r}, t0={t0})")
        for t in range(50):
            # Математична модель з 04_growth_theory_resilience.md
            noise = random.uniform(-0.05, 0.05)
            self.metrics["latency"] = K / (1 + np.exp(-r * (t - t0))) + noise
            print(f"  t={t} | Затримка (Latency): {self.metrics['latency']:.3f}s")
            time.sleep(0.5)

    def trigger_cardinality_explosion(self, n_labels=1000):
        """Імітація вибуху індексу TSDB через унікальні User-ID."""
        print(f"[*] Injecting High Cardinality noise into metrics...")
        for i in range(n_labels):
            user_id = f"user_{random.getrandbits(32)}"
            # Емуляція відправки метрики в Prometheus Pushgateway
            # requests.post(..., labels={"user_id": user_id}) 
            pass
        print(f"[!] Injected {n_labels} unique labels. TSDB OOM risk increased.")

if __name__ == "__main__":
    chaos = ChaosMachine()
    # Запуск сценарію розпаду затримки
    chaos.simulate_logistic_latency_decay()

```

---

## 3. Математика Хаосу: Моделювання інтенсивності відмов

При створенні синтетичних аномалій важливо дотримуватися ймовірнісного розподілу. Ми використовуємо **розподіл Пуассона** для генерації частоти вхідних запитів та **розподіл Парето** для моделювання "важких хвостів" (Long Tail) у затримках.

Ймовірність виникнення  помилок за інтервал часу  обчислюється як:

Де  — інтенсивність помилок (error rate).

### Завдання для лаби:

1. **Ін'єкція помилок:** Запустіть Chaos Agent для генерації 5-хвилинного логістичного розпаду продуктивності.
2. **Детекція:** Перевірте, чи спрацює ваш предиктивний алерт на основі точки перегину  до того, як затримка досягне  порогу.
3. **Байєсівська фільтрація:** Налаштуйте фільтр так, щоб відсікти "шумові" сплески CPU (тривалістю < 5с), але не пропустити реальний лінійний Memory Leak.

---

## 4. Hidden Risks та Техніка безпеки

* **Blast Radius (Радіус ураження):** Ніколи не запускайте Chaos Agent у продакшн-середовищі без обмеження ресурсів на рівні `cgroups` (Resource Quotas у Kubernetes). Агент може "вбити" не лише цільовий сервіс, а й сусідні поди через Resource Starvation.
* **Cleanup:** Переконайтеся, що ваш скрипт має механізм "Stop-the-noise". У разі втрати зв'язку з агентом, система може залишитися в деградованому стані назавжди.
* **Кореляція часу:** При аналізі результатів хаос-тесту обов'язково синхронізуйте таймстемпи ін'єкцій з логами Loki для точної перевірки BERT-класифікатора.

