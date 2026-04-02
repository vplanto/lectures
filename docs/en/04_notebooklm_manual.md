# NotebookLM: Complete Guide (Audio, Deep Research & Deep Reading)

> 🇺🇦 [Українська версія](../04_notebooklm_manual.md) | 🇬🇧 English version

**Context:** Google's NotebookLM tool has evolved. It's no longer just a "reader" but a full-fledged operating system for knowledge work.

**Guide objective:** Teach you to use three operational modes:

1. **Passive Mode (Podcasts):** Converting materials into audio overviews.
2. **Active Mode (Deep Research):** Autonomous agent-driven information retrieval.
3. **Storage Mode (Deep Reading):** Organizing a "Second Brain" for complex documentation.

---

## Part 1. Passive Mode (Podcasts)
AI will retell your materials in a dialogue format.

### Key Feature
As of 2025, NotebookLM supports **Audio Overviews** in Ukrainian.

### Step-by-Step Instructions

#### Step 1: Language Setup (Most Important!)
1. Open [NotebookLM](https://notebooklm.google/).
2. Click the **Settings** icon (⚙️).
3. Select **"Output Language" → Ukrainian**.

#### Step 2: Create a Notebook and Add Sources
**What you can upload:**

* **PDF / Google Docs:** Research papers, reports.
* **YouTube:** Links to lectures (reads transcripts).
* **Text:** Copied text from clipboard.

#### Step 3: Generation and Customization
Click **"Customize"** before generating.

**Example prompts:**

* *For learning:* "Explain this as if for a first-year student. Use simple analogies."
* *For debates:* "One host defends the technology, the other harshly criticizes its security."

#### Step 4: Consumption
Click **"Generate"**, wait 5 minutes, download the MP3, and listen offline.

---

## Part 2. Deep Research (Autonomous Researcher)
This is an agent that goes online, scans hundreds of sources, and creates a "distilled" report.

### Action Algorithm:
1. Click the **Deep Research** button.
2. Enter a topic (e.g., *"Comparative analysis of Web3 architectures: Solana vs Ethereum in 2025"*).
3. **Plan:** Adjust the plan proposed by the agent.
4. **Execution:** Wait 5-10 minutes while the agent "googles."
5. **Result:** Receive a structured **"Research Overview"** document.

### Use Cases
* **The "Blank Page Killer":** Entering a new topic from scratch (obtaining terminology and bibliography).
* **The "Gap Filler":** Finding what's missing from your lectures (e.g., recent critiques of methods).

---

## Part 3. Workflow: "Closed Loop"
**Scenario: Preparing for a Complex Exam**

1. **Deep Research:** *"Find the latest use cases for Java Spring Boot in 2025"*.
2. **Local Context:** Upload this report + your lectures to the notebook.
3. **Audio Overview:** *"Generate a dialogue where hosts discuss how theory from my lectures applies to these cases"*.
4. **Listen:** Listen to the podcast on your way to university.

---

## Part 4. Case Study: "Hacking" the Reading List (Deep Reading)
Engineers often face a problem: they need to master 1000+ pages of technical documentation (e.g., **Kubernetes Documentation** or **C++ Standards**). Reading this linearly is impossible. Remembering everything is unrealistic.

Here NotebookLM works as an **"Interactive Index"**.

### How to Set Up a "Technical Second Brain":
Instead of reading one PDF at a time, we create a **Meta-Notebook**.

1. **Create Notebook:** Name it, for example, *"Kubernetes Expert"*.
2. **Input (Bulk Upload):** Upload:
   * Official documentation (PDF).
   * "Kubernetes Up and Running" book.
   * Your internal corporate instructions (Wiki).
   * RFCs and Best Practices from GitHub.

3. **Limit:** Remember the limit (50 sources per notebook). Combine small files into one PDF before uploading.

### Use Cases (Engineering Use Cases):

#### A. Unified Search
Instead of searching separately in documentation, separately in books, and separately in Google:

* **Query:** *"How to configure Ingress Controller for gRPC traffic according to our internal security policies?"*
* **Answer:** The system will find technical instructions in official docs, check your internal Security Policies (which you uploaded), and provide a ready config.

#### B. Conflict Detection
You've uploaded old architecture and new requirements.

* **Query:** *"Are there conflicts between network settings in `legacy_config.pdf` and new requirements in `security_2025.pdf`?"*
* **Result:** NotebookLM will point to specific items that contradict each other.

#### C. On-demand FAQ
You forgot a command parameter.

* **Query:** *"What's the difference between `Recreate` and `RollingUpdate` deployment strategies? Give me a comparison table."*
* **Result:** Instant table based on *your* sources, without ChatGPT hallucinations.

> **Conclusion:** You don't stop reading documentation. But now you have an assistant that "remembers" every page of a thousand and can instantly find connections you missed.

---

## Final Checklist
* [ ] **Language:** Settings verified (Output Language → Ukrainian).
* [ ] **Sources:** At least 2-3 files uploaded or Deep Research executed.
* [ ] **Context:** If it's technical documentation — verify PDFs aren't outdated.
* [ ] **Prompt:** Role specified for hosts (e.g., "Criticize", "Explain").
* [ ] **Offline:** Audio downloaded to phone.
* [ ] **Fact-Checking:** Remember that even Deep Research can be wrong. Always follow citations.

---

## Update: December 2025. Agentic Workflows
**Insight source:** Stanford CS230 (Andrew Ng), Q4 2025.

Andrew Ng notes that the main trend of 2025 is the shift from Chatbot (Zero-shot) to **Agentic Workflows**. The "Deep Research" feature in NotebookLM is your first accessible **Autonomous Agent**.

### 1. Role Shift: From "Reader" to "Manager"
Previously you consumed content. Now you manage an agent that processes it. This changes the workflow to a classic management cycle:

1.  **Intent:** You don't just write a query, you set a TOR (Technical Specification).
    * *Bad Intent:* "Find something about Kubernetes."
    * *Good Intent:* "Create a comparison table of Cilium and Calico network policies for high-load clusters, focus on latency."
2.  **Tool Use (Execution):** The agent (Deep Research) plans the search, reads sources, discards noise, and synthesizes a report.
3.  **Audit:** The most important stage. You verify not the "style" of text, but **citations**.

### 2. The "Lazy Manager" Trap
Like with "Vibe Coding", there's a risk of **"Vibe Research"**.
* **Risk:** You blindly trust the agent's report (Deep Research) without opening primary sources.
* **Reality:** Agents can "hallucinate" facts or take quotes out of context.
* **Rule:** If the report is used for critical decisions (architecture, money, health), you **must** click on [Citation] and verify the original.

### 3. Productivity Thesis
According to the lecture, AI "magic" disappears, leaving "mundane utility". NotebookLM is not magic. It's a tool for **scaling your reading**.
* **Before:** You could read 3 articles per hour.
* **With agent:** You can "scan" 50 articles per hour and deeply read the 3 most important ones the agent found.
