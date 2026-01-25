# Modernizing Educational Materials with AI Assistants

> 🇺🇦 [Українська версія](../00_method_changes.md) | 🇬🇧 English version

**Dear colleagues!**

Today I want to share an experience that, in my opinion, could fundamentally change our approach to methodological work.

### Trigger for Change: Honest Feedback

It all started with a simple but very accurate phrase in student feedback on one of my courses: **"it was difficult, we understood little"**. This comment made me think. We often use the same presentations for years, which seem clear to us, but for today's first-year student they can be dry, overloaded, and disconnected from reality. I realized that something needed to change, but how to find the time and fresh ideas?

### New Tool: AI as a Methodological Assistant

My solution was to engage modern AI assistants, specifically Google's Gemini. It's important to understand: I use it not as a "term paper generator," but as a **methodological sparring partner**. The key advantage of this model is its huge "memory window" (up to 1 million tokens), which allows it to hold in context not just one question, but entire textbooks and our dialogues over many weeks.

### What has Changed in Essence? "Before" and "After"

Practically, I'll show you two versions of materials, but here are the key differences:

| ❌ **"Before": Old Presentation** | ✅ **"After": New Markdown Workshop** |
| :--- | :--- |
| **Static document.** Difficult to make changes, need to update slides, save PDF. | **"Living" text document.** Easy to edit, supplement, fix errors "on the fly". |
| **Theoretical focus.** List of dry facts and definitions. | **Practical focus.** Built around solving an interesting, modern task (e.g., "Monobank Banking" simulator). |
| **Disconnection from reality.** Abstract examples that don't always resonate with students. | **Relevance.** Tasks that use youth-oriented and understandable ideas for Ukrainian students. |
| **"Dead" format.** Students can only view slides. | **Interactive format.** Contains questions for the group, tasks for joint discussion, checkpoints for self-verification. |
| **Fighting cheating.** Attempts to ban phones/internet.| **AI-Resistant Assessment.** Architecture defense tasks and live-coding, where AI is just a calculator.|

### Philosophical Pivot: From "Coder" to "Engineer"

Based on the experience of leading physics and mathematics schools and the requirements of the high-load industry, we are changing the very positioning of programming in our courses.

  * **Code is the new calculator.** The ability to write loops is no longer a unique advantage; AI does that. The unique advantage becomes **fundamental understanding** of *why* this code works the way it does (Memory Layout, Complexity, Networking basics).
  * **Problem > Syntax.** We reject the "code works" assessment. We move to "problem solved efficiently" assessment. If a student wrote working code but can't explain why they chose `LinkedList` instead of `ArrayList` for a cache — that's a failure.
  * **Architecture defense.** Instead of dry tests, we introduce the practice of "solution defense" (analogous to RFC in Big Tech). The student must argue tool selection based on precise metrics, not "it's more convenient for me."

-----

### New Toolset: Markdown and GitHub

This transformation became possible thanks to the transition to a new, much simpler set of tools.

#### 1\. Markdown (MD) — self-formatting text

**What is it?** This is a simple markup language that allows you to format text using plain symbols. Instead of selecting font and size in PowerPoint, you simply write:

```markdown
# Header
* This will be a list item
**And this is bold text**
```

**Why is it important?** This allows you to completely **separate content from form**. You focus on the text and lecture structure, not slide design. This incredibly speeds up work.

#### 2\. GitHub — home for our materials

**What is it?** Most know GitHub as a place to store code. But it's an ideal tool for storing our lectures in Markdown format too.

**Why is it important?**

  * **Version control:** You get a "time machine" for your materials. You can always see who made changes and when, and return to any previous version.
  * **Centralized repository:** All course materials are in one place, accessible from any computer.

#### 3\. GitHub Pages — your personal educational site in 5 minutes

**What is it?** This is a free GitHub feature that **instantly converts** your Markdown files into a simple but elegant website, accessible to all students via one link.

**Why is it important?** This is the embodiment of the principle **"final product, not a diploma"**. Instead of a set of scattered files, you create a unified, professional educational resource for your course.

-----

### Level 1: How to Start?

**This is accessible to everyone:** Google provides **free access to Gemini Advanced for 1 year** for all teachers and scholars in Ukraine.

**Universal prompt:** So you don't start from scratch, here's a prompt that will let you immediately set the right tone for working with the assistant.

> Imagine you are my methodological assistant, an expert in pedagogical design and in my discipline [discipline name]. Our goal is to help me transform an outdated lecture into a modern, interactive workshop for first-year students.
>
> We will follow two main ideas:
>
> 1.  **Engineering thinking, not just code:** Every workshop should teach not only "how to write," but also "why exactly this way." The student should understand the cost of their solution (memory, speed).
> 2.  **Focus on people for the country:** Examples and tasks should be relevant and, when possible, show how students can bring benefit here and now.
>
> I will provide you with my old materials and ideas, and you will structure them, suggest improvements, generate interesting tasks, and ask clarifying questions so that together we create the best possible educational material. Shall we begin?

### Level 2: Stress-Testing Materials (Validation)

When a draft workshop or lecture is already created, it's important to remove the "rose-colored glasses." For this, we change the assistant's role from "helper" to "harsh opponent." This allows identifying weaknesses, logical holes, and boring moments before students do.

Use this prompt to check already-prepared ideas ("Brutally Honest Advisor" mode):

> **Validation prompt:**
>
> "From now on, stop 'convenient interlocutor' mode. Act as my brutally honest high-level advisor.
>
> Test my thinking for strength. Question my assumptions. Identify my 'blind spots.' No flattery. No smoothing corners.
>
> If my argumentation is weak — decompose it and point out errors. If I'm avoiding uncomfortable questions — point it out directly. Show where I'm making excuses or underestimating risks.
>
> After critical analysis, give me a precise, prioritized action plan to reach the next level. Hide nothing."

**How to use:**

1.  Upload the previously created Markdown lecture file.
2.  Activate this prompt.
3.  Ask: *"Analyze this material. Does this example really explain the essence, or am I just filling airtime? Where will students lose attention?"*

Thank you for your attention. I'm confident that this approach can significantly improve the quality of our methodological work and increase student engagement.
