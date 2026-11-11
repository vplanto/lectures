# Інженерний бриф: Архітектура довіреного контексту в AI SDLC та сценарії взаємодії ролей

> 🇺🇦 Українська версія | ⬅️ [Назад до огляду розділу AI SDLC](./index.md) | ➡️ [План науково-практичної статті](./02_context_architecture_plan.md)

**Тема:** Довірений контекст у AI SDLC (`Git + Jira + Confluence + TestRail + Infra/APM`)  
**Критерії проектування:** Прозорість · Простота · Точність

---

## 1. Нативні канали ролей, локальні зони та відповідальні артефакти

Кожна команда працює у своєму нативному інструменті, має власну локальну кухню (яка деталізує реалізацію, але **не повинна суперечити** загальному декларативному стану) і відповідає за свій канонічний артефакт:

| Роль | Нативний канал роботи | Локальний контекст *(Вимога: не суперечити спільній істині)* | Відповідальний артефакт *(Зона володіння)* | Специфіка та парадокси в AI SDLC |
| :--- | :--- | :--- | :--- | :--- |
| **DEV + Coding Agents** | Git, IDE, Jira *(не працюють у Confluence напряму, бо це ненадійний канал без гілкового версіонування)* | Локальна підготовка контексту (`.claude/`, `research.md`, `plan.md`), гілки коду, юніт-тести | Код, OpenAPI/Proto-контракти, схеми БД, Helm/Infra-маніфести, локальні інваріанти сервісу | Збирати контекст легко для нової фічі з нуля, але вкрай важко для **масштабного багфікса чи Change Request (CR)**, де треба підняти історію старих рішень |
| **QA + QA Agents** | База ТК (**TestRail**), Confluence, Jira | Локальні тестові дані, автотести, внутрішні чеклісти QA | Тест-кейси та сьюти у TestRail, матриці станів та **граничні кейси (*edge cases*)** у Confluence/Jira | **Парадокс ізоляції TestRail від Dev:** це і *бага*, і *фіча* залежно від зрілості процесу (див. розд. 3.1) |
| **Perf + Perf Agents** | Confluence, Jira, Інфраструктура (k8s/Grafana/APM) | Скрипти навантаження (JMeter/Gatling/k6), локальні профілі та сирі метрики | Паспорт **Infra Sizing** + **NFR** (SLA, p99, RPS/CPM) у Confluence/Jira | Ізольовано Perf не знає бізнес-логіки нової фічі й як її коректно навантажити. **ШІ стає Enabler-ом**, розмиваючи вузьку спеціалізацію і даючи Perf-у доступ до суті фічі |
| **PO** | Виключно Jira та Confluence | Пріоритизація, клієнтські запити | Епіки/Stories/CR у Jira (`FixVersion`), реєстр бізнес-фіч у Confluence | Єдина точка легалізації зовнішніх вимог у систему інженерного контексту |
| **BA / Arch** | Чати, Email, робочі зустрічі | Чорнові схеми, листування, обговорення компромісів | **Фінальний артефакт — завжди зміна в Confluence/Jira** (бізнес-правила, ADR до 200 рядків, *rejected* варіанти) | **Правило неіснування:** усе, що залишилося в чатах чи пошті і не лягло в Confluence/Jira, для агентів і команд не існує |
| **External 3rd-Party** | Email, зовнішній Confluence/Jira або портал | Зовнішні інтеграційні системи | Запит або узгодження контракту *(якщо прийшло в Email — **обов'язок PO** перенести це в Jira/Confluence)* | Споживають тільки верифікований та очищений (*sanitized*) зовнішній зріз без внутрішньої кухні |
| **AppSec / SRE** *(дод.)* | Сканери, алерти, Jira, Confluence | Логи безпеки та інцидентів, post-mortems | Threat Models / Runbooks під поточний продакшн-реліз | Потребують точного зрізу стану **`As-Is`** на проді без домішок майбутніх чернеток |

---

## 2. Природа інструментів: Jira як Потік Подій vs Confluence як Декларативний Стан

1. **Jira = Event Stream (Труба з подіями змін):**
   - Jira описує не поточний стан системи, а **потік подій (дельт)**: *Story, Bug, Change Request, Epic*.
   - **Архітектурний патерн «Алгоритми на трубі подій»:** на життєвий цикл Jira-подій (створення тікета, зміна статусу, прив'язка `FixVersion`, коментар людини, закриття релізу) підписуються спеціалізовані агентні обробники (пайплайни), які трансформують подію у зміну декларативного стану або збирають контекст під конкретний тип події.
2. **Confluence = Declarative State з явним версіонуванням `As-Is` та `To-Be`:**
   - Щоб Confluence перестав бути «звалищем без версій», вводиться жорстке правило декларації станів: кожна сторінка/секція фіксує поточний стан продакшену (**`As-Is [Release v2.4]`**) та окремо маркований блок цільових змін (**`To-Be [Target v2.5 / Jira-Key]`**), який вливається в основний стан `As-Is` лише після фактичного виходу релізу.
3. **Вимога до локального контексту команд (Consistency Contract):**
   - Кожна команда має право на свої локальні артефакти (Dev у Git, QA у TestRail, Perf у своїх скриптах), але діє жорстке правило: **локальний контекст деталізує реалізацію, проте не може суперечити спільному декларативному стану в Confluence/Jira**.

---

## 3. Чотири глибокі інженерні дилеми (Нюанси реальної практики)

### 3.1. Недоступність TestRail для Dev — це Фіча чи Бага? (Залежність від зрілості)
* **Коли це ФІЧА (Незалежний контур верифікації / Double-Blind Black-Box Guardrail):** якщо Dev або його кодинг-агент бачить точні покрокові тест-кейси та тестові дані з TestRail наперед, виникає ефект *«teaching to the test» (підгонка коду під тест)* — агент пише вузькі `if-else` суворо під відомі сценарії QA, пропускаючи системну доменну логіку. Коли ж QA будує кейси в TestRail незалежно від коду (за спільними бізнес-вимогами), виникає подвійна незалежна перевірка (*Double-Blind Verification*), яка ловить хибні припущення розробника.
* **Коли це БАГА (Пізнє виявлення очевидних граничних умов):** якщо Dev взагалі не знає, які *класи граничних умов* (наприклад, порожні списки, невалідні валюти, ідемпотентні повтори, гонки запитів) QA буде перевіряти, елементарні дефекти спливають занадто пізно і породжують дорогий цикл повернень тікетів.
* **Баланс зрілості:** Dev-у потрібні не сирі тест-кейси з TestRail, а **декларативний перелік граничних умов (Edge-Case Matrix / Rules)** у спільному контексті на вході, тоді як конкретні тестові сценарії та пейлоади в TestRail залишаються незалежною зброєю QA.

### 3.2. Ізоляція Performance-інженерів та ШІ як Enabler (Розмивання меж спеціалізації)
* **Проблема:** Perf-інженер глибоко знає інфраструктуру (k8s, пули БД, профілювання JVM/Go) та NFR, але через вузьку спеціалізацію часто відірваний від внутрішньої бізнес-логіки нових фіч і не знає, *як саме функціонально правильно* зімітувати складний користувацький сценарій.
* **Роль ШІ як Enabler-а:** наявність структурованого контексту фічі (від BA/Dev/QA) дозволяє Perf-інженеру за допомогою AI-агента миттєво зрозуміти бізнес-флоу, згенерувати валідні пейлоади та сценарії навантаження і перетворити свою роботу з «тестування сферичного ендпоінта у вакуумі» на реалістичне тестування бізнес-фічі. Водночас і Dev за допомогою ШІ отримує доступ до експертизи Perf (паспорта Infra Sizing та NFR-бюджетів) ще на етапі планування коду.

### 3.3. Мала зміна vs Масштабна зміна + Керованість людиною (Human Override)
* **Мала зміна (Small Bug / Story — T-Shirt S) — режим «Inline All-in-One» з контролем людини:** агент на трубі Jira автоматично збагачує тікет коротким резюме на 4 рядки (яке правило зачеплено, вплив на API, граничні умови, перевірка NFR), **але людина має простий механізм прямого впливу (Human-in-the-Loop Override)** — інженер/PO/QA може прямо в коментарі чи полі тікета відкоригувати припущення агента («Ні, тут правило X не діє, фіксимо за сценарієм Y»), і агент зобов'язаний перебудувати контекст суворо за цим рішенням людини.
* **Масштабна зміна — розділення за типами задач:**
  - *Нова фіча (Greenfield Feature):* легко розкладається з Jira Epic по чистих нових розрізах у Confluence (Capability, ADR, QA Edges, Perf NFR) і легко збирається у чистий пакет для Dev.
  - *Масштабний багфікс або наскрізний Change Request (Brownfield CR / Deep Refactoring):* **найважчий випадок для збірки контексту Dev-ом!** Бо зміна ріже по живому десятки старих фіч, де історія розмазана по сотнях старих тікетів Jira, ADR та комітів. Саме тут ручний збір контексту гальмує команду, і потрібен спеціалізований алгоритм зворотного збору імпакту (*Impact & History Archeology Agent* на базі гібридного пошуку `BM25 + Vector DB` по Git + Jira + Confluence).

---

## 4. Вибір технологій у цій картині (Фінальна формула)

1. **Людські першоджерела за зонами відповідальності:** `Jira` (труба подій) + `Confluence` (декларативний стан з версіями `As-Is / To-Be`) + `TestRail` (база ТК QA) + `Code Git` (код, контракти, Infra-конфіги).
2. **Алгоритми на трубі подій Jira (Event-Driven Context Agents):**
   - реагують на події в Jira (Нова фіча / Change Request / Масштабний баг / Реліз);
   - перевіряють цілісність контексту (*Context Linter*) та враховують ручні правки людини (*Human Override* у Jira/Confluence);
   - збирають із федеративних джерел верифікований **Context Bundle** у локальне середовище Dev-а (`research.md -> plan.md`) або генерують *Release Delta* (`v2.4 -> v2.5`) для QA і Performance.
3. **Похідний гібридний індекс (`BM25 + Vector DB`) + Процедурна пам'ять (`CLAUDE.md` / `MEMORY.md`):**
   - `BM25 + Vector DB` не замінюють Confluence/Git як першоджерело, а працюють як **пошуковий двигун для складних Change Requests та масштабних багфіксів**, коли агенту треба швидко знайти історичний імпакт по старих тікетах і сторінках із жорсткою фільтрацією за актуальною версією;
   - екстрактивні сервіси (`Mem0`) — тільки для особистих преференцій інженера; графові (`Zep/Graphiti`) — опціонально для трасування складних міжсервісних залежностей при CR.

---

## 5. Сценарії та діаграми послідовності (Sequence Diagrams + PlantUML у згорнутих секціях)

### 5.1. Загальна наскрізна діаграма: Jira Event Pipe → Declarative State (Confluence) → Context Assembly за ролями

![Загальна наскрізна діаграма потоків контексту](./diagrams/01_overall_context_flow.png)

<details>
<summary><b>Показати вихідний код PlantUML (01_overall_context_flow)</b></summary>

```plantuml
@startuml 01_overall_context_flow
autonumber
skinparam BoxPadding 10
skinparam ParticipantPadding 8

box "Зовнішній світ та Бізнес" #F9F9F9
  actor "3rd-Party" as Ext
  actor "PO" as PO
  actor "BA / Arch" as BA_Arch
end box

box "Спільна Шина Подій та Декларативний Стан" #EEF5FF
  participant "Jira\n(Event Stream / Труба подій)" as Jira
  participant "Event Pipe Agents\n(Алгоритми на трубі)" as PipeAgent
  participant "Confluence\n(Declarative State:\nAs-Is v2.4 | To-Be v2.5)" as Conf
  database "Hybrid Index\n(BM25 + Vector DB)" as Index
end box

box "Інженерні Команди та їхні Локальні Зони (Silos)" #F2FFF2
  actor "QA + QA Agent" as QA
  database "TestRail\n(База ТК)" as TestRail
  actor "Perf + Perf Agent" as Perf
  database "Infra / NFR\n(k8s / APM / Load)" as Infra
  actor "DEV + Coding Agent" as Dev
  database "Code Git & Local Context\n(.claude / plan.md)" as Git
end box

== 1. Ініціація та легалізація контексту (PO / BA / Arch / 3rd-Party) ==
Ext -> PO : Запит по Email / каналу комунікації
PO -> Jira : Створює подію (Epic / Story / CR, FixVersion=v2.5)
BA_Arch -> Conf : Оновлює Declarative State (секція To-Be v2.5:\nCapabilities, ADR <= 200 рядків, Rejected options)
Conf -> Index : Фонова переіндексація з тегом версії (To-Be v2.5)

== 2. Збагачення фасетами QA та Performance ==
Jira -> PipeAgent : Webhook: Нова подія зміни (v2.5)
PipeAgent -> QA : Тригер: визначити класи граничних умов
QA -> Conf : Декларує Edge-Case Rules (To-Be v2.5)
QA -> TestRail : Створює детальні тест-кейси (ізольовано від Dev)
PipeAgent -> Perf : Тригер: оцінити NFR та Infra impact (AI як Enabler пояснює бізнес-флоу)
Perf -> Conf : Фіксує NFR Budgets (p99, RPS) та Infra Sizing (To-Be v2.5)

== 3. Збірка контексту для DEV та Імплементація ==
Dev -> PipeAgent : Беру в роботу таску Jira (v2.5)
PipeAgent -> Conf : Читає As-Is (v2.4) + To-Be (v2.5): Capabilities, ADR, Edge-Rules, NFR/Infra
PipeAgent -> Index : Пошук історичного імпакту (для CR / складних багів)
PipeAgent -> Git : Генерує локальний Context Bundle (research.md -> plan.md)
Dev -> Git : Human Review плану (200 рядків) -> Кодування + Контракти (OpenAPI) + PR

== 4. Верифікація та Реліз (Перехід To-Be -> As-Is) ==
Git -> PipeAgent : PR Merged / Release Candidate v2.5
PipeAgent -> QA : Генерує QA Release Diff (v2.4 -> v2.5)
QA -> TestRail : Прогін тест-сьютів (Black-Box верифікація)
PipeAgent -> Perf : Генерує Perf Release Delta (SQL/API/Infra diff + бізнес-сценарій)
Perf -> Infra : Навантажувальний тест за профілем фічі
PO -> Jira : Release v2.5 Shipped
Jira -> PipeAgent : Подія Release v2.5
PipeAgent -> Conf : Промоутить блоки To-Be (v2.5) у канонічний стан As-Is (v2.5)
PipeAgent -> Ext : Публікує зовнішній зріз контрактів (Sanitized v2.5)
@enduml
```

</details>

---

### 5.2. Сценарій 1: PO + BA + Arch + 3rd-Party (Перетворення чатів і листів на версіонований Declarative State)

![Сценарій 1: PO + BA + Arch + 3rd-Party](./diagrams/02_po_ba_arch_3rdparty_flow.png)

<details>
<summary><b>Показати вихідний код PlantUML (02_po_ba_arch_3rdparty_flow)</b></summary>

```plantuml
@startuml 02_po_ba_arch_3rdparty_flow
autonumber
actor "External 3rd-Party" as Ext
actor "Product Owner (PO)" as PO
actor "BA / Architect" as Arch
participant "Chats / Email\n(Неформальний шум)" as Chat
participant "Jira\n(Event Stream)" as Jira
participant "Confluence\n(Declarative State)" as Conf
participant "Context Linter Agent\n(Алгоритм на трубі)" as Agent

Ext -> Chat : Лист: "Змініть ліміт API та додайте вебхук"
note right of Chat : Поки це в Email/чаті —\nдля системи контексту цього НЕ існує
Chat -> PO : PO опрацьовує вхідний запит
PO -> Jira : Створює Change Request (CR-402, FixVersion=v2.5)
PO -> Arch : Запит на проробку бізнес-правил та архітектури

Arch <-> Chat : Обговорення компромісів у чатах / на дзвінках
Arch -> Conf : Оновлює сторінку фічі та створює ADR (<= 200 рядків):\n- Блок [As-Is v2.4] не затирається!\n- Додається блок [To-Be v2.5 | CR-402]\n- Фіксуються [Rejected Options] (чому відкинули варіант Б)
Arch -> Jira : Прикріплює посилання на To-Be секцію Conf та переводить у "Ready"

Jira -> Agent : Подія: статус "Ready for Dev"
Agent -> Conf : Перевіряє цілісність: чи є ADR, чи розділено As-Is / To-Be
alt Виявлено суперечність зі старими правилами
  Agent -> Jira : Коментар-попередження для BA/Arch/PO (Conflict Alert)
  PO -> Conf : Уточнює правило (Human Override)
else Контекст консистентний
  Agent -> Jira : Маркує CR-402 як "Context Verified"
end
@enduml
```

</details>

---

### 5.3. Сценарій 2: DEV + Coding Agent (Три режими: Мала зміна vs Нова фіча vs Масштабний Багфікс / CR)

![Сценарій 2: DEV + Coding Agent — три режими збірки контексту](./diagrams/03_dev_three_modes_flow.png)

<details>
<summary><b>Показати вихідний код PlantUML (03_dev_three_modes_flow)</b></summary>

```plantuml
@startuml 03_dev_three_modes_flow
autonumber
actor "Developer" as Dev
participant "Jira\n(Event Stream)" as Jira
participant "Event Pipe Agent\n(Context Assembler)" as PipeAgent
database "Hybrid Index\n(BM25 + Vector DB)" as Index
participant "Confluence\n(ADR, NFR, Edge-Rules)" as Conf
participant "Local Dev Env & Git\n(.claude / research.md / plan.md)" as LocalGit

== Варіант А: Мала зміна (Small Bug / Story — T-Shirt S, "Inline All-in-One") ==
Jira -> PipeAgent : Подія: Small Bug/Story створено
PipeAgent -> Conf : Швидкий пошук зачепленого правила
PipeAgent -> Jira : Записує прямо в тікет 4 рядки:\n[Rule + API Impact + Edge Cases + NFR]
Dev -> Jira : Бачить неточність агента -> править коментарем/полем (Human Override!)
Jira -> PipeAgent : Подія: Human Override
PipeAgent -> LocalGit : Оновлює локальний контекст таски суворо за правкою людини

== Варіант Б: Нова Фіча (Greenfield Epic — T-Shirt M/L) ==
Dev -> PipeAgent : Старт роботи над Story з нового Epic (FixVersion=v2.5)
PipeAgent -> Jira : Читає маніфест Epic-а (прямі лінки на розрізи)
PipeAgent -> Conf : Стягує чисті блоки [To-Be v2.5]: Capability + ADR + QA Edge-Rules + Perf NFR/Infra
PipeAgent -> LocalGit : Формує чистий Context Bundle (вікно < 40%, без історичного сміття)

== Варіант В: Масштабний Багфікс або Наскрізний Change Request (Brownfield CR — Найважчий кейс!) ==
Dev -> PipeAgent : Старт роботи над масштабним CR / Deep Bugfix
note right of PipeAgent : Прямих лінків в одному Epic недостатньо —\nзміна ріже десятки старих правил!
PipeAgent -> Index : Археологія імпакту (BM25 за кодами/методами + Vector за семантикою)
Index --> PipeAgent : Повертає шляхи до зачеплених сторінок Conf, старих ADR та Jira-багів
PipeAgent -> Conf : Вичитує актуальні секції [As-Is v2.4] та перевіряє [Rejected Options]
PipeAgent -> LocalGit : Складає Impact Report (`research.md`): список зачеплених модулів,\nінфраструктурних лімітів (DB pools/queues) та колізій
Dev -> LocalGit : Верифікує `research.md` -> генерує `plan.md` (200 рядків) -> пише код і тести
LocalGit -> Conf : При зміні контракту чи локального правила — сигналізує про потребу апдейту To-Be
@enduml
```

</details>

---

### 5.4. Сценарій 3: QA (Парадокс TestRail: декларативні граничні умови для Dev vs ізольовані ТК у TestRail)

![Сценарій 3: QA та парадокс TestRail](./diagrams/04_qa_testrail_flow.png)

<details>
<summary><b>Показати вихідний код PlantUML (04_qa_testrail_flow)</b></summary>

```plantuml
@startuml 04_qa_testrail_flow
autonumber
actor "QA Engineer\n+ QA Agent" as QA
participant "Jira\n(Event Stream)" as Jira
participant "Confluence\n(Спільний Declarative State)" as Conf
database "TestRail\n(Локальна база ТК QA)" as TestRail
actor "DEV\n+ Coding Agent" as Dev

== Фаза 1: Проектування (Shift-Left — що віддати Dev, а що сховати в TestRail) ==
Jira -> QA : Подія: Нова фіча / CR у роботі (v2.5)
QA -> Conf : Публікує Декларативні Граничні Умови (Edge-Case Matrix):\n(ліміти полів, переходи станів, коди помилок)\n[Видимо для DEV — щоб уникнути пізніх багів]
QA -> TestRail : Генерує детальні тестові сценарії, конкретні пейлоади та тест-дані\n[Ізольовано в TestRail — щоб агент DEV не робив "підгонку під тест"]

Conf -> Dev : Агент DEV забирає Edge-Case Matrix у свій `plan.md` і враховує в коді
note over TestRail, Dev : DEV не читає сирі кроки TestRail ->\nзберігається незалежна Black-Box верифікація

== Фаза 2: Перевірка релізу (Release Diff) ==
Jira -> QA : Збірка релізу v2.5 готова до тестування
Conf -> QA : Алгоритм генерує QA Release Diff (що змінилося між v2.4 та v2.5 у правилах і API)
QA -> TestRail : Оновлює регресійний сьют у TestRail на базі Release Diff та проганяє тести
@enduml
```

</details>

---

### 5.5. Сценарій 4: Performance Engineer (ШІ як Enabler, що з'єднує бізнес-фічу з Infra + NFR)

![Сценарій 4: Performance Engineer та ШІ як Enabler](./diagrams/05_perf_enabler_flow.png)

<details>
<summary><b>Показати вихідний код PlantUML (05_perf_enabler_flow)</b></summary>

```plantuml
@startuml 05_perf_enabler_flow
autonumber
actor "Perf Engineer\n+ Perf Agent" as Perf
participant "Jira\n(Event Stream)" as Jira
participant "Confluence\n(Capabilities + Infra/NFR)" as Conf
participant "Code Git\n(OpenAPI / SQL / Helm)" as Git
actor "DEV\n+ Coding Agent" as Dev
database "Load Env & APM\n(k8s / Gatling / Grafana)" as LoadEnv

== Фаза 1: Передача Infra та NFR обмежень у контекст DEV ==
Perf -> Conf : Декларує актуальний паспорт Infra Sizing (k8s limits, DB pools, Kafka)\nта NFR Budgets (p99 latency, peak RPS) для сервісів
Conf -> Dev : Агент DEV підтягує Infra + NFR у локальний контекст перед написанням коду\n(щоб код не поклав пул БД на проді)

== Фаза 2: Підготовка навантажувального тестування (AI як Enabler для Perf) ==
Git -> Jira : Зміни коду влито в гілку релізу v2.5
Jira -> Perf : Тригер навантажувального тестування v2.5
note right of Perf : Проблема: Perf знає інфраструктуру,\nале не знає внутрішньої бізнес-логіки нової фічі
Perf -> Conf : Perf-агент читає бізнес-опис фічі (від BA) та граничні стани (від QA)
Perf -> Git : Perf-агент аналізує Технічну Дельту (нові SQL-запити, індекси, API, зміни в Helm)
Conf --> Perf : AI Enabler синтезує для Perf-інженера зрозумілий сценарій:\n"Як правильно функціонально прокачати фічу + які пейлоади використати"
Perf -> LoadEnv : Генерує скрипти навантаження (k6/Gatling) і проганяє тест на стенді
LoadEnv -> Conf : Оновлює фактичні метрики продуктивності релізу v2.5 (As-Is Baseline)
@enduml
```

</details>
