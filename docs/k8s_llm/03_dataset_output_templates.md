# Шаблони Еталонних Відповідей (Output)

Цей документ визначає **"Gold Standard"** (Золотий Стандарт) відповідей, яких ми очікуємо від моделі.
Для кожного запису в датасеті поле `output` має бути валідним JSON-об'єктом.

---

## 1. Структура JSON (Schema)

```json
{
  "summary": "Стислий опис ситуації для людини (Human-readable summary)",
  "root_cause": "Технічний ID проблеми з Таксономії (напр., NodeOvercommit)",
  "recommendation": "Інженерна дія з конкретними числами або іменами параметрів"
}
````

-----

## 2\. Шаблони для Функціональних Відмов (Fixing Breakages)

### 1\. Проблеми Запуску (Startup)

#### 🔴 Сценарій: Image Pull Issues (NEW)

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The Pod failed to pull the container image due to an authentication error or incorrect tag.",
  "root_cause": "ImagePullBackOff",
  "recommendation": "Verify the image tag spelling. If using a private registry, ensure a valid Secret is created and referenced in `imagePullSecrets`."
}
```

#### 🔴 Сценарій: Config Dependencies (NEW)

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The container cannot start because it fails to mount a required ConfigMap or Secret volume.",
  "root_cause": "CreateContainerConfigError",
  "recommendation": "Create the missing ConfigMap named 'app-settings' in the same namespace, or correct the reference in the Pod spec."
}
```

#### 🔴 Сценарій: CrashLoopBackOff

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The application container is failing to start properly, causing repeated restarts. Logs indicate a missing configuration environment variable.",
  "root_cause": "CrashLoopBackOff",
  "recommendation": "Check the container logs for the 'DB_URL' variable error. Ensure the 'DB_URL' environment variable is defined in the Deployment spec."
}
```

### 2\. Проблеми Планування (Scheduling)

#### 🔴 Сценарій: Resource Shortage (Pending)

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The Pod cannot be scheduled because no node has enough free CPU capacity to satisfy the request of 2000m (2 Cores).",
  "root_cause": "ResourceShortage",
  "recommendation": "Option 1: Enable Cluster Autoscaler to provision a new node. Option 2: Reduce CPU requests if 2000m exceeds actual requirements."
}
```

#### 🔴 Сценарій: Affinity/Taint Conflicts (NEW)

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The scheduler cannot place the Pod because existing nodes have Taints that the Pod does not Tolerate, or Affinity rules are too restrictive.",
  "root_cause": "PlacementConstraintConflict",
  "recommendation": "Add a Toleration for key 'gpu-node' to the Pod spec, or remove the Taint from the target node."
}
```

### 3\. Проблеми Виконання (Runtime)

#### 🔴 Сценарій: OOMKilled

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The container was terminated with Exit Code 137 (OOMKilled), meaning it tried to use more memory than its limit.",
  "root_cause": "OOMKilled",
  "recommendation": "Increase the memory limit to at least **384Mi** (current 256Mi + 50% buffer). Check logs for potential memory leaks."
}
```

#### 🔴 Сценарій: Probe Failures (NEW)

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The container is running but fails Liveness/Readiness probes, causing restarts or traffic removal.",
  "root_cause": "ProbeFailure",
  "recommendation": "Increase `initialDelaySeconds` to 30s (currently 5s) to allow the slow application to startup fully before checks begin."
}
```

-----

## 3\. Шаблони для FinOps та Оптимізації (Optimization)

### 4\. Інфраструктурна Ефективність

#### 🟢 Сценарій: Node Resource Overcommit (NEW)

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The sum of CPU limits of all pods on this node exceeds the physical node capacity by 150%. This creates a high risk of CPU throttling during peak loads.",
  "root_cause": "NodeResourceOvercommit",
  "recommendation": "Rebalance the cluster by adding a new node or reducing limits for non-critical (BestEffort) workloads."
}
```

#### 🟢 Сценарій: Resource Fragmentation (Bin Packing)

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The cluster has sufficient total capacity (20 CPU free), but resources are fragmented across nodes, preventing the scheduling of a 4 CPU pod.",
  "root_cause": "ResourceFragmentation",
  "recommendation": "Use a **Descheduler** policy to evict smaller pods and consolidate them onto fewer nodes to create a large slot."
}
```

#### 🟢 Сценарій: Infrastructure Mismatch

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "Node resource usage is unbalanced: Memory is highly utilized (90%) while CPU is idle (10%).",
  "root_cause": "InfrastructureMismatch",
  "recommendation": "Migrate this workload to a **Memory Optimized** instance family (e.g., AWS **r5** or **r6g**)."
}
```

#### 🟢 Сценарій: Legacy Storage Class

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The PVC uses the legacy 'gp2' StorageClass.",
  "root_cause": "LegacyStorageClass",
  "recommendation": "Migrate the StorageClass to **gp3** to reduce costs by ~20% and decouple IOPS from size."
}
```

### 5\. Конфігураційна Оптимізація

#### 🟢 Сценарій: Quantitative Right-Sizing

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The Pod is overprovisioned. It requests 4000m CPU but uses only ~500m at peak (P95).",
  "root_cause": "OverprovisionedRequests",
  "recommendation": "Reduce CPU requests from 4000m to **600m**. Formula: P95 Usage (500m) + 20% Buffer (100m)."
}
```

#### 🟢 Сценарій: Missing Limits (NEW)

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "The Pod has no resource limits defined (QoS Class: BestEffort). It can consume all available node resources, starving other pods.",
  "root_cause": "MissingLimits",
  "recommendation": "Set resource limits (e.g., memory: 512Mi) to prevent the 'Noisy Neighbor' effect."
}
```

#### 🟢 Сценарій: VPA Recommendation

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "Workload exhibits high variability (50m to 1200m). Static requests are inefficient.",
  "root_cause": "DynamicWorkloadInefficiency",
  "recommendation": "Deploy a **Vertical Pod Autoscaler (VPA)** in 'Auto' mode."
}
```

### 6\. Надійність (Reliability)

#### 🟠 Сценарій: Critical Workload QoS Risk

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "Critical DB workload is running with 'Burstable' QoS (requests < limits).",
  "root_cause": "CriticalWorkloadQoSRisk",
  "recommendation": "Set memory requests equal to limits to achieve QoS class **Guaranteed**."
}
```

#### 🟠 Сценарій: Single AZ Risk

  * **Output Template:**

<!-- end list -->

```json
{
  "summary": "All replicas are in the same Availability Zone.",
  "root_cause": "SingleAvailabilityZoneRisk",
  "recommendation": "Configure **TopologySpreadConstraints** to distribute pods across zones."
}
```