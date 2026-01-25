# "Epsilon-Greedy" Principle: How to Avoid Mental Overfitting

> 🇺🇦 [Українська версія](../07_mental_overfitting.md) | 🇬🇧 English version

In machine learning there's a concept of **Overfitting**: when a model perfectly memorizes training data but becomes helpless on real, new data. It minimizes error in familiar environment, getting stuck in **Local Minimum**.

The same happens with engineers. Social media algorithms and convenient tools (LLM) drive us into a "warm bath" where we only hear what we agree with and solve tasks only with familiar methods.

---

##  Principle Essence

Imagine an agent in Reinforcement Learning. If it always chooses the best known action (Exploitation), it will never learn that the adjacent path, which initially looks worse, leads to **Global Maximum**.

To survive, the agent uses **$\epsilon$-Greedy** strategy:
* With probability $1 - \epsilon$ it does what's profitable (works as usual).
* With probability $\epsilon$ (e.g., 10-20%) it makes a **random, illogical step** (Exploration).

**Golden rule:** To grow, you must artificially introduce "noise" and discomfort into your information diet.

---

## How Does This Concern Engineers?

### Situation 1: The Stack Zealot (Technology Fanaticism)
You've been writing Java for 10 years. You follow Java bloggers, attend Java conferences.
* **Overfitting:** You try to solve "Cold Start" problem in Serverless with JVM methods, because that's all you know.
* **Epsilon-Greedy action:** Spend time analyzing Go or Rust idioms, even if "Java is better." This forces brain to build new neural connections.

### Situation 2: The "Yes-Bot" Loop (Working with AI)
You use ChatGPT/Claude to generate ideas. AI (configured for "help") always agrees with your architecture, just "polishing" it.
* **Overfitting:** You lose critical thinking and criticism processing skills. Your architecture becomes brittle.
* **Epsilon-Greedy action:** Use prompt: *"Find 3 critical vulnerabilities in this design. Be ruthless. Imagine you're a Senior Architect who hates this approach."*

### Situation 3: Information Bubble
YouTube/LinkedIn algorithms serve you content you "like."
* **Overfitting:** You stop seeing alternative viewpoints. Your worldview narrows to a point.
* **Epsilon-Greedy action:** Read sources that irritate you. Use **"Steelmanning"** method: try to build maximally strong argument *in favor* of technology or idea you despise.

---

##  Safety Checks

Exploration is expensive. Don't turn "horizon broadening" into procrastination.

1.  **Timeboxing (10% Rule):** If you work 40 hours, maximum 4 hours go to "Exploration." No more. If there's no insight in this time — kill process.
2.  **Force Analogy:** You are *obligated* to find connection. Read about ant architecture? Find 3 common features with your microservice (e.g., decentralized management). Didn't find — you just wasted time.
3.  **Low Friction:** Don't need to write pet project in Rust to understand Rust. It's enough to read article "Why Rust Ownership is hard" and try to explain it to yourself. Minimize context switching cost.

-----

## Action Algorithm (Cognitive Hygiene Checklist)

1.  **Inject Noise:** Once a week consume content orthogonal to your profession. This is source of metaphors for engineering solutions (Serendipity).
2.  **Pull, don't Push:** Reject algorithmic feeds in favor of RSS or direct visits to resources. Don't let algorithm decide for you.
3.  **Human RNG:** Talk to person not from IT. Their "naive" questions can destroy your overcomplicated logic faster than Code Review.
4.  **Increase Temperature:** If you're too comfortable and clear — you're not learning. You're just "exploiting" cache. Do something where you feel like a beginner.
