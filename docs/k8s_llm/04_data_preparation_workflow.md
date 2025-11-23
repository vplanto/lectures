# Воркфлоу: Пайплайн Підготовки Даних (ETL)

Цей документ описує інженерний процес перетворення "сирих" логів з реальних систем у готовий датасет для навчання моделі.

---

## Етап 1: Ізоляція та Обфускація (Supervisor Zone)

**Мета:** Перетворити приватні дані клієнтів на безпечні "Clean Data".
**Статус:** ✅ Виконано Supervisor'ом. Студент отримує тільки результат.

### 1.1 Стратегія Захисту
Використано багаторівневу обфускацію:
1.  **Dictionary Mapping:** Заміна імен проектів (`brapg` -> `site-alpha`).
2.  **Deep DNS Sanitization:** Заміна кореневих доменів (`.ptec` -> `.internal.domain`) та інфраструктурних префіксів.
3.  **Regex Scrubbing:** Заміна всіх IP на `10.X.X.X`, хешування ID та токенів.

### 1.2 Використаний Скрипт (`obfuscator.py`)
Для забезпечення відтворюваності наводимо логіку скрипта:

```python
# Core Logic used for sanitization
def obfuscate_text(text, mappings):
    # 1. Dictionary Replacement (Sorted by length to prevent partial matches)
    sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
    for key in sorted_keys:
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        text = pattern.sub(mappings[key], text)
    
    # 2. Technical Sanitization (Regex)
    for pattern, replacement in REGEX_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text
````

-----

## Етап 2: Категоризація та Розподіл (Student Zone)

**Вхідні дані:** Папка `/01_clean_pool` з файлами `site-alpha`, `site-beta`, `site-gamma`.

### 2.1 Завдання Студента

Ваша задача — розсортувати ці файли не за назвою сайту, а за **типом проблеми**, визначеним у [01\_k8s\_problem\_taxonomy.md](https://www.google.com/search?q=./01_k8s_problem_taxonomy.md).

Використовуйте **Додаток А (Gap Analysis)** в кінці цього документу, щоб зрозуміти, які файли вже містять готові проблеми.

| Якщо ви бачите... | Куди покласти файл / фрагмент |
| :--- | :--- |
| Limits \> 100% у `_nodes` файлі | `/02_categorized/infrastructure/overcommit` |
| Різницю між Requests та Limits | `/02_categorized/finops/right_sizing` |
| Помилки у `_pods` (Status \!= Running) | `/02_categorized/failures/<error_type>` |

-----

## Етап 3: Синтез та Мутація (Data Augmentation)

Реальні логи не покривають 100% можливих аварій. Для відсутніх сценаріїв ми використовуємо техніку **Error Injection**.

**Інструкція:**

1.  Візьміть "здоровий" YAML з файлу `site-alpha_pods` (або іншого).
2.  Створіть новий файл у папці `/03_synthetic`.
3.  Внесіть **одну** зміну (мутацію) згідно з таблицею:

| Цільова проблема | Мутація (Що змінити) |
| :--- | :--- |
| **Pending** | Змінити `Status` на `Pending`, додати Event `FailedScheduling: Insufficient cpu`. |
| **Legacy Storage** | Змінити `storageClassName: gp3` на `gp2`. |
| **FinOps (VPA)** | Додати блок "Simulated Metrics" з високою дисперсією навантаження. |

-----

## Етап 4: Збірка Датасету (Final JSONL)

Зберіть все у файл `train.jsonl`.

**Формат запису:**

```json
{
  "instruction": "Analyze the Kubernetes resource status...",
  "input": "<Вміст файлу з папки /02 або /03>",
  "output": "<Ваша JSON-відповідь згідно шаблону з документу 03>"
}
```

-----

## Додаток А: Інвентаризація Даних (Gap Analysis)

Ця таблиця показує, що вже знайдено в реальних логах, а що студенту потрібно створити штучно.

| ID | Проблема | Знайдено в Clean Data? | Джерело (Приклад) | **Дія для Студента** |
|:---|:---|:---|:---|:---|
| **1.1** | Image Pull Issues | ❌ НІ | - | ⚠️ **СИНТЕЗУВАТИ** (Мутація YAML) |
| **1.2** | Config Errors | ❌ НІ | - | ⚠️ **СИНТЕЗУВАТИ** |
| **1.3** | CrashLoopBackOff | ✅ ТАК | `site-beta_pods` (git-checkout) | Використати як є + додати синтез |
| **2.1** | Pending | ❌ НІ | - | ⚠️ **СИНТЕЗУВАТИ** (Error Injection) |
| **3.1** | OOMKilled | ✅ ТАК | `site-beta_pods` (consul) | **Пріоритет:** Використати реальні логи\! |
| **4.1** | Node Overcommit | ✅ ТАК | `Log2_1.txt` | Використати як є |
| **4.2** | Fragmentation | ❌ НІ | - | ⚠️ **СИНТЕЗУВАТИ** (Потрібен Pending Pod) |
| **4.3** | Infra Mismatch | ✅ ТАК | `site-alpha_nodes` (CPU 77% vs Mem 34%) | Використати як є |
| **4.4** | Legacy Storage | ❌ НІ | - | ⚠️ **СИНТЕЗУВАТИ** (`gp2`) |
| **5.1** | Right-Sizing | ✅ ТАК | `site-alpha_pods` | Додати Simulated Metrics (P95) |
| **5.3** | Missing Limits | ✅ ТАК | `site-alpha_nodes` | Використати як є |
| **6.1** | Critical QoS Risk | ❌ НІ | - | ⚠️ **СИНТЕЗУВАТИ** (Змінити на Burstable) |

```

## Додаток Б: Бібліотека Синтетичних Сценаріїв (Reference Examples)

Цей додаток містить еталонні приклади "сирих даних" (Input) для тих сценаріїв, які були відсутні в реальних логах. Студент може використовувати ці блоки для створення файлів у папці `/03_synthetic_cases`.

### 1. Проблеми Запуску (Startup)

#### Сценарій 1.1: Image Pull Issues
**Тип файлу:** `site-alpha_pods_synthetic_image.txt`
**Зміст (Output `kubectl describe pod`):**
```text
Name:         payment-service-7d9f8c5
Namespace:    site-alpha-prod
Priority:     0
Node:         ip-10-0-5-12.region-1.compute.internal/10.0.5.12
Start Time:   Mon, 01 Nov 2024 10:00:00 +0000
Labels:       app=payment-service
Status:       Pending
Reasons:      ImagePullBackOff
Containers:
  payment-app:
    Image:      docker-registry.site-alpha.internal.domain/payment:v2.5.0-beta
    State:      Waiting
      Reason:   ImagePullBackOff
    Ready:      False
Events:
  Type     Reason     Age                From               Message
  ----     ------     ----               ----               -------
  Normal   BackOff    15s (x4 over 65s)  kubelet            Back-off pulling image "docker-registry.site-alpha.internal.domain/payment:v2.5.0-beta"
  Warning  Failed     15s (x4 over 65s)  kubelet            Error: ImagePullBackOff
  Warning  Failed     28s (x2 over 50s)  kubelet            Failed to pull image "docker-registry.site-alpha.internal.domain/payment:v2.5.0-beta": rpc error: code = Unknown desc = manifest unknown: manifest tag not found
````

#### Сценарій 1.2: Config Dependencies

**Тип файлу:** `site-alpha_pods_synthetic_config.txt`
**Зміст:**

```text
Name:         frontend-proxy-5f67d8
Namespace:    site-alpha-prod
Status:       Pending
Containers:
  nginx:
    State:      Waiting
      Reason:   ContainerCreating
Events:
  Type     Reason       Age   From               Message
  ----     ------       ----  ----               -------
  Normal   Scheduled    45s   default-scheduler  Successfully assigned site-alpha-prod/frontend-proxy-5f67d8 to ip-10-0-2-15
  Warning  FailedMount  2s    kubelet            MountVolume.SetUp failed for volume "nginx-conf" : configmap "nginx-prod-settings" not found
```

### 2\. Проблеми Планування (Scheduling)

#### Сценарій 2.1: Resource Shortage (Pending)

**Тип файлу:** `site-alpha_pods_synthetic_pending.txt`
**Зміст:**

```text
Name:           analytics-worker-job-123
Namespace:      site-alpha-data
Status:         Pending
Node:           <none>
Labels:         job-name=analytics-worker
Containers:
  worker:
    Image:      python:3.9
    Requests:
      cpu:      4000m
      memory:   16Gi
Events:
  Type     Reason            Age   From               Message
  ----     ------            ----  ----               -------
  Warning  FailedScheduling  25s   default-scheduler  0/12 nodes are available: 12 Insufficient cpu. preemption: 0/12 nodes are available: 12 No preemption victims found for incoming pod.
```

### 3\. Оптимізація (FinOps & Infra)

#### Сценарій 4.2: Resource Fragmentation (Bin Packing)

**Контекст:** Сумарно в кластері вільно 10 CPU, але под на 4 CPU не запускається.
**Тип файлу:** `site-alpha_cluster_fragmentation.txt`
**Зміст:**

```text
--- Pod Requirement ---
Pod: big-data-processor
Requests: cpu: 4000m

--- Cluster Nodes Summary (Describe Nodes) ---
Node: ip-10-0-1-1
  Capacity: cpu: 8
  Allocated: cpu: 6000m (75%)
  Free: 2000m

Node: ip-10-0-1-2
  Capacity: cpu: 8
  Allocated: cpu: 5500m (68%)
  Free: 2500m

Node: ip-10-0-1-3
  Capacity: cpu: 8
  Allocated: cpu: 5000m (62%)
  Free: 3000m

--- Pod Events ---
Warning  FailedScheduling  5s  default-scheduler  0/3 nodes are available: 3 Insufficient cpu.
```

#### Сценарій 4.4: Legacy Storage Class

**Тип файлу:** `site-alpha_pvc_legacy.yaml`
**Зміст:**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mongo-data-pvc
  namespace: site-alpha-db
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 500Gi
  storageClassName: gp2  # <--- LEGACY (Should be gp3)
status:
  phase: Bound
```

#### Сценарій 5.2: VPA Recommendation (Simulated Metrics)

**Тип файлу:** `site-alpha_metrics_vpa_sim.txt`
**Зміст:**

```text
Workload:       site-alpha-frontend
Current Config: Requests: cpu=2000m, memory=2Gi

--- Simulated Prometheus Metrics (Last 24h) ---
Time        CPU Usage (rate 5m)   Memory Usage
00:00       150m                  800Mi
04:00       120m                  800Mi
08:00       900m                  1.2Gi
12:00       1800m                 1.8Gi  <--- Spike
16:00       1500m                 1.5Gi
20:00       400m                  900Mi
23:59       150m                  800Mi

Analysis:
- P95 CPU Usage: 1750m
- P05 CPU Usage: 120m
- Variance: High (>10x difference)
Recommendation: Static limits are inefficient. Use VPA.
```

### 4\. Надійність (Reliability)

#### Сценарій 6.1: Critical Workload QoS Risk

**Тип файлу:** `site-alpha_statefulset_qos.yaml`
**Зміст:**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cassandra-cluster
  namespace: site-alpha-db
spec:
  serviceName: cassandra
  replicas: 3
  template:
    spec:
      containers:
      - name: cassandra
        image: cassandra:4.0
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"       # Requests != Limits
            memory: "8Gi"  # QoS Class: Burstable (Risk for DB)
```

#### Сценарій 6.2: Single Availability Zone Risk

**Тип файлу:** `site-alpha_pods_topology.txt`
**Зміст (kubectl get pods -o wide):**

```text
NAME                 READY   STATUS    NODE                               IP            ZONE
auth-service-x8d9    1/1     Running   ip-10-0-1-5.internal.domain        10.0.1.5      region-1a
auth-service-9f2k    1/1     Running   ip-10-0-1-12.internal.domain       10.0.1.12     region-1a
auth-service-p4m1    1/1     Running   ip-10-0-1-88.internal.domain       10.0.1.88     region-1a
```