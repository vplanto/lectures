---
title: "02 Detection And Symptoms"
type: lecture
module: Module 2
prerequisites: module 1
layout: default
---

> **Академічна доброчесність.** Матеріали відповідають вимогам [Закону України № 4742-IX](../DISCLAIMER.md). Використання ШІ — [протокол](../10_ai_lectures.md). Оцінювання — [Risk & Reward](../06_grading_experiment.md). Джерела курсу: [sources.md](./sources.md).

# Довідник: Як виявити проблему (Команди та Симптоми)

Цей документ визначає, які саме технічні дані (Input) потрібно зібрати для кожного типу проблеми.
**Важливо:** Для задач оптимізації (FinOps) ми використовуємо не лише миттєві знімки (`kubectl`), а й історичні метрики (**Prometheus P95/P99**), оскільки це єдиний спосіб дати обґрунтовану рекомендацію.

---

## А. Функціональні Відмови (Functional Failures)

### 1. Проблеми Запуску (Startup Phase)

#### 1.1 Image Pull Issues
* **Команда:** `kubectl describe pod <pod-name>`
* **Маркер проблеми:** Статус `ImagePullBackOff` або `ErrImagePull`.
* **Синтетичний приклад (Input):**
    ```text
    State:          Waiting
      Reason:       ImagePullBackOff
    Events:
      Type     Reason     Message
      ----     ------     -------
      Warning  Failed     Failed to pull image "my-app:v2": rpc error: code = Unknown desc = manifest unknown
    ```

#### 1.2 Config Dependencies
* **Команда:** `kubectl describe pod <pod-name>`
* **Маркер проблеми:** Помилка монтування тому або конфігу.
* **Синтетичний приклад (Input):**
    ```text
    State:          Waiting
      Reason:       CreateContainerConfigError
    Events:
      Warning  FailedMount  MountVolume.SetUp failed: configmap "app-prod-config" not found
    ```

#### 1.3 CrashLoopBackOff
* **Команда:** `kubectl describe pod` + `kubectl logs --previous`
* **Маркер проблеми:** Часті перезапуски, Exit Code != 0.
* **Синтетичний приклад (Input):**
    ```text
    State:          Waiting
      Reason:       CrashLoopBackOff
    Last State:     Terminated
      Reason:       Error
      Exit Code:    1
    --- Logs ---
    Panic: database connection url is empty. Check DB_URL env variable.
    ```

### 2. Проблеми Планування (Scheduling Phase)

#### 2.1 Resource Shortage (Pending)
* **Команда:** `kubectl describe pod <pod-name>`
* **Маркер проблеми:** `FailedScheduling` через `Insufficient cpu/memory`.
* **Синтетичний приклад (Input):**
    ```text
    Status:       Pending
    Events:
      Warning  FailedScheduling  0/10 nodes are available: 10 Insufficient cpu.
    ```

#### 2.2 Affinity/Taint Conflicts
* **Команда:** `kubectl describe pod <pod-name>`
* **Маркер проблеми:** Конфлікт правил розміщення.
* **Синтетичний приклад (Input):**
    ```text
    Status:       Pending
    Events:
      Warning  FailedScheduling  0/5 nodes are available: 2 node(s) had untolerated taint {gpu: true}, 3 node(s) didn't match PodAntiAffinity rules.
    ```

### 3. Проблеми Виконання (Runtime Phase)

#### 3.1 OOMKilled (Memory Limit)
* **Команда:** `kubectl describe pod <pod-name>`
* **Маркер проблеми:** `Reason: OOMKilled`, `Exit Code: 137`.
* **Синтетичний приклад (Input):**
    ```text
    State:          Terminated
      Reason:       OOMKilled
      Exit Code:    137
    Limits:
      memory:  256Mi
    ```

#### 3.2 Probe Failures
* **Команда:** `kubectl describe pod <pod-name>`
* **Маркер проблеми:** `Unhealthy` Liveness/Readiness проба.
* **Синтетичний приклад (Input):**
    ```text
    Events:
      Warning  Unhealthy  Liveness probe failed: Get "[http://10.244.0.5:8080/health](http://10.244.0.5:8080/health)": context deadline exceeded
    ```

---

## Б. Неефективність та Оптимізація (Optimization & FinOps)

### 4. Інфраструктурна Ефективність

#### 4.1 Node Resource Overcommit
* **Команда:** `kubectl describe node <node-name>`
* **Маркер проблеми:** Сума лімітів значно перевищує 100% (Allocated resources).
* **Синтетичний приклад (Input):**
    ```text
    Node: ip-10-0-5-4
    Allocated resources:
      Resource           Limits
      --------           ------
      cpu                6000m (150%)  <-- Risk of throttling
      memory             20Gi  (125%)  <-- Risk of OOM
    ```

#### 4.2 Resource Fragmentation (Bin Packing)
* **Команда:** `kubectl describe nodes` (summary) + `kubectl describe pod <pending-pod>`
* **Маркер проблеми:** Сумарно в кластері місця багато, але под `Pending` бо жодна окрема нода не має достатньо великого шматка.
* **Синтетичний приклад (Input):**
    ```text
    Pod Request: 4 CPU
    Cluster Capacity: 20 CPU Free (Total)
    Node 1: 2 CPU Free
    Node 2: 1.5 CPU Free
    Node 3: 2.5 CPU Free
    Event: 0/3 nodes are available: 3 Insufficient cpu.
    ```

#### 4.3 Infrastructure Mismatch
* **Команда:** `kubectl describe node` (Allocated requests).
* **Маркер проблеми:** Диспропорція (напр., RAM повна, CPU пустий).
* **Синтетичний приклад (Input):**
    ```text
    Instance Type: m5.2xlarge (8 vCPU, 32GiB)
    Allocated requests:
      cpu:    800m (10%)
      memory: 28Gi (87%)
    ```

#### 4.4 Legacy Storage Class
* **Команда:** `kubectl get pvc -o yaml`
* **Маркер проблеми:** Використання `gp2` замість `gp3`.
* **Синтетичний приклад (Input):**
    ```yaml
    kind: PersistentVolumeClaim
    spec:
      storageClassName: gp2  # Legacy
      resources:
        requests:
          storage: 100Gi
    ```

### 5. Конфігураційна Оптимізація (Metrics-Driven)

#### 5.1 Quantitative Right-Sizing (Overprovisioning)
* **Команда:** **Simulated Prometheus Query** (на заміну `kubectl top`).
* **Маркер проблеми:** `Requests` значно більші за `P95 Usage`.
* **Синтетичний приклад (Input):**
    ```text
    Pod: payment-service-v1
    --- Configuration ---
    Requests:
      cpu: 2000m (2 Cores)
      memory: 4Gi
    --- Prometheus Metrics (7d range) ---
    cpu_usage_p95: 150m
    mem_usage_p95: 300Mi
    ```
    *(Тут видно, що запит 2000m, а реальне пікове споживання лише 150m. Рекомендація має базуватися на 150m + буфер).*

#### 5.2 VPA Recommendation (Dynamic Workload)
* **Команда:** **Simulated Prometheus Metrics** (Time Series).
* **Маркер проблеми:** Висока дисперсія навантаження (вдень високо, вночі низько), статичні ліміти неефективні.
* **Синтетичний приклад (Input):**
    ```text
    Pod: frontend-app
    --- Configuration ---
    Requests: cpu: 1000m
    --- Usage History (24h) ---
    00:00 - 08:00: Avg 50m
    08:00 - 18:00: Avg 950m (Spikes to 1200m)
    18:00 - 23:59: Avg 100m
    ```
    *(Статичний request 1000m буде марнувати ресурси вночі. Потрібен VPA).*

#### 5.3 Missing Limits
* **Команда:** `kubectl get pod -o yaml`
* **Маркер проблеми:** Відсутність поля `limits`.
* **Синтетичний приклад (Input):**
    ```yaml
    resources:
      requests:
        cpu: 100m
      # Limits are missing -> QoS BestEffort/Burstable
    ```

### 6. Архітектурна Надійність

#### 6.1 Critical Workload QoS Risk
* **Команда:** `kubectl get pod -o yaml` + Context "Database".
* **Маркер проблеми:** Критичний сервіс (БД) має `requests` < `limits` (Burstable), що створює ризик тротлінгу.
* **Синтетичний приклад (Input):**
    ```yaml
    kind: StatefulSet
    metadata:
      name: postgres-db
    spec:
      resources:
        requests:
          memory: 1Gi
        limits:
          memory: 4Gi  # Burstable QoS is risky for Production DB
    ```

#### 6.2 Single Availability Zone Risk
* **Команда:** `kubectl get pods -o wide`
* **Маркер проблеми:** Всі репліки в одній зоні.
* **Синтетичний приклад (Input):**
    ```text
    NAME     NODE-ZONE
    pod-1    eu-central-1a
    pod-2    eu-central-1a
    pod-3    eu-central-1a
    ```