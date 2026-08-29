**Autonomous Machine Learning Research Agent for Recommender Systems**

In response to some queries, our engineers have provided updates to the problem statement to improve clarity and to support participants better.  
**Problem Statement last updated: 25 August 2026, 9:10PM.**  
**Technical Workshop Webinar with Q\&A** will be held on **28 Aug, 2:00 to 2:45pm.**  
Click here to join the webinar\!

1. **Background**

### **Motivation**

Machine learning engineers (MLEs) spend much of their time on a single activity: **taking a dataset and a set of metrics, then iterating on a model again and again to push the score higher.** This work is inherently cyclic — every round repeats the same loop, shown in Figure 1\.  
![][image1]  
> **Figure 1\. The MLE iteration loop.** A closed cycle of five core stages, plus a reflection step that feeds the next round:

> 1. **Read the problem** — understand the given dataset and the target metrics.  
> 2. **Inspect data** — study data distribution through exploratory data analysis (EDA).  
> 3. **Engineer features** — build and select input features (see Appendix A.5).  
> 4. **Train \+ tune** — choose a model, set the loss function, and tune hyperparameters.  
> 5. **Evaluate** — read the metrics, check for overfitting, and consult the leaderboard.

> The result of the **evaluate** stage drives a **reflect \+ revise** step, which decides what to change and loops back into the next iteration — re-inspecting the data and adjusting the features. The cycle repeats until the score plateaus.  
Two of these stages — **engineer features** and **train \+ tune** — are carried out almost entirely in code: the engineer writes scripts to transform the data, define the model, and run training. In other words, each turn of the loop produces and modifies code. This is what makes the loop a natural target for automation: it is structured and repeatable, yet writing and revising that code is exactly the kind of task a code-generating LLM can take on.  
The loop is also repetitive and mechanical. It draws heavily on "engineering intuition," but many individual steps are well-structured and repeatedly exercised in practice — which is precisely why automating the whole cycle has become an active research direction.

### **Prior Work**

Over the past two years, a new line of work has set out to automate this loop: the **Autonomous ML Research Agent**, an LLM-driven agent that runs the cycle in Figure 1 on its own. It reads the problem, **writes the code** for each stage, trains and evaluates the model, reflects on the results, revises its approach, and finally produces a submission. Representative systems include:

* **MLE-Bench** \[1\] (OpenAI) — a benchmark of 75 Kaggle competitions, now a standard evaluation suite for such agents.  
* **AIDE** \[2\] (Weco AI) — a state-of-the-art agent that frames ML engineering as code optimization and explores the space of solutions via tree search.  
* **AI-Scientist-v2** \[3\] (Sakana AI) — an end-to-end agent for autonomous scientific and ML research, using agentic tree search to form hypotheses, run experiments, and write up results.

### **This Challenge**

This challenge asks participants to design an **autonomous ML research agent**. Given a public ML dataset and a set of metrics, the agent must **autonomously** run the full loop of Figure 1 — read the problem, engineer features, train and tune the model, evaluate, then reflect and iterate — to reach the highest possible score across the test sets. Writing the code for each stage is part of the agent's job, not something provided in advance.  
> **New to recommender systems?** All benchmarks in this challenge come from the recommendation domain (the KuaiRand family). If terms such as CTR, multi-task learning, NDCG, or Recall@K are unfamiliar, start with the **Appendix: A Primer on Recommender Systems** . At the end of this document — a concept map plus an annotated reading list designed to get you oriented in 1–2 hours.

2. **Problem Statement**

### **The Task**

Design and implement an Autonomous ML Research Agent. For each benchmark, the agent must autonomously:

1. **Reproduce the official baseline.** Stand up a working end-to-end pipeline and confirm it reaches the official baseline's reported validation score. (The official baseline is a fixed, organizer-provided reference — see *Benchmarks*. Any starter pipeline the agent builds for itself is an internal step, not the reference it is scored against.)  
2. **Iterate on the pipeline.** Autonomously draw on established methods from both industry and academia to improve each stage of the pipeline (see Figure 1), and apply those improvements in code. The agent develops using **only the training split and the public validation feedback** — it never has access to the hidden test set.  
3. **Improve over the baseline.** Through repeated iterations, drive the **validation** score above the official baseline. Improvement need not be strictly monotonic — as with real-world data, the trajectory may fluctuate — but the agent should show a clear, sustained ability to keep improving relative to the baseline. Final ranking is computed once, on the **hidden test set**, using the submission the agent designates as final.

### **Task Requirements**

1. **Runs end-to-end and aims to beat the baseline.** The agent must run the full pipeline on the required benchmark (KuaiRand-Pure) and reach a converged result; attempting the bonus benchmark (KuaiRand-1k & KuaiRand-27k) is optional. The target is a hidden-test score that exceeds the official baseline; the actual delta achieved — positive or negative — is what feeds into the Primary metric scoring (see Judging Criteria), so falling short of the baseline is scored continuously rather than treated as a disqualifying failure.  
2. **Iterates autonomously across the full stack.** The agent should improve the solution on its own, driven by its own evaluation of results. Improvements may target any part of the algorithmic stack — not just the model architecture, but every upstream and downstream module is fair game. The goal is to **minimize human intervention** — a fully autonomous run is the ideal, but a well-instrumented **semi-automated** pipeline that requires only a handful of interventions is an acceptable and realistic outcome; in practice, we measure how little human intervention a run requires (e.g. the number of manual interventions).  
3. **Robust operation.** The pipeline should run reliably with **minimal human intervention**. Robustness here is about how the agent *handles* difficulty, not how often it succeeds — we do not score it by failure count, since a capable agent may fail only on genuinely hard problems. What matters is that when a step fails (a code error, a timeout, an unexpected input), the agent can recover, retry, or route around it, and that long iterative runs neither crash, stall, nor diverge.  
3. **Constraints & Scope**

| Category | Constraints & Scope Details |
| :---- | :---- |
| In scope | Any open-source library or framework (PyTorch, RecBole, TorchRec, LightGBM, …) Any papers, public solutions, or pretrained weights Changes to any pipeline stage — not just the model |
| Out of scope | No external training data or pretrained weights trained on these benchmarks' test labels No hidden-test access during development (train \+ validation only) |
| Limits | **KuaiRand-Pure**: NDCG@10 / Recall@50, click \= positive (fixed) (*Required*); **KuaiRand-1k** & **KuaiRand-27k**: same task and metrics (*Bonus*) Hidden test scored once, on the final submission Compute budget: *TBD* |
| Allowed assumptions | Fixed `train / validation / hidden-test` split per dataset Official baseline, scores & evaluation script (incl. convergence rule) Example submission \+ output schema |

4.   
   **Available Resources & Data**

### **Starter Kit**

To lower the barrier to entry — especially for participants new to recommender systems — the challenge provides a standard starting point:

1. **Fixed data splits**: Kuairand itself provides data splits according to dates (`log_standard_4_08_to_4_21_*.csv` & `log_standard_4_22_to_5_08_*.csv`) you can use 4/08–4/21 standard → train, 4/22–5/08 standard first 50% → validation, 4/22–5/08 standard last 50% → test. Teams develop on train \+ validation only, and evaluate the performance on test set.  
2. **Official baseline**: a fixed, organizer-provided reference pipeline per dataset (refer to [CWM](https://github.com/hyz20/CWM?utm_source=chatgpt.com) for Kuairand), with its baseline scores published. Beating *this* baseline is what counts — not a baseline the team builds itself.  
3. **Evaluation script**: the exact scoring code (NDCG@K / Recall@K for KuaiRand), plus the convergence rule (ε and *N*) and the absolute-delta aggregation. Refer to [CWM](https://github.com/hyz20/CWM?utm_source=chatgpt.com) for Kuairand.  
4. **Submission format**: a minimal, runnable example submission and the required output schema.  
5. **Run-log requirements**: each iteration should record its **hypothesis**, the **code diff**, the resulting **metrics**, and any **error / recovery events**. These logs are how judges assess **Autonomy** (scored under Impact & Relevance) and **Robustness** (scored under Technical Execution) — see Judging Criteria.  
6. **LLM coding agent**: you can use whatever you like, or use [Trae](https://www.trae.ai/pricing) from ByteDance, which provides "Limited offer: new user 7-day free trial".

### **Benchmarks**

**KuaiRand-Pure is required** and determines 100% of the primary score. **KuaiRand-1k and KuaiRand-27k are bonus datasets** — attempting them is optional and earns extra credit, but neither is required to complete the primary score.  
**Resource policy.** This is a hackathon, so external resources are open by default: use any open-source library (PyTorch, RecBole, TorchRec, LightGBM, …), read any papers, docs, or public solutions, and use pretrained model weights freely. The agent is expected to draw on whatever published methods it can find — that is what makes it a *research* agent.  
There is **one hard rule: no external training data.** Training must rely only on the KuaiRand datasets listed below — no augmenting, joining, or pre-training on any other dataset, and no pretrained model whose weights were trained on these benchmarks' test labels. This single rule is what keeps the hidden-test ranking fair; everything else is unrestricted.

| Dataset | Domain & Description | Metrics | Scale |
| :---- | :---- | :---- | :---- |
| **KuaiRand** (Kuaishou) Three released variants: **KuaiRand-Pure** is required, while **KuaiRand-1k** and **KuaiRand-27k** are bonus. | Short-video feed. 12 feedback signals (click / like / follow / comment / forward / long\_view / play\_time …) plus a randomized-exposure intervention that supports counterfactual evaluation. **Relevance label and K are fixed by the organizers** (see Starter Kit / TBD): the default task treats `click` as the positive relevance label and reports **NDCG@10 / Recall@50**. The exact label definition and K values are pinned in the Starter Kit so every team solves the same task. | NDCG@10 / Recall@50  | Pure: 1.4M interactions (27K users × 7.6K items). 1k: 11.7M. 27k: 322M. |

Links: KuaiRand — [https://kuairand.com](https://kuairand.com/)  
> KuaiRand's randomized-exposure data also enables off-policy / counterfactual evaluation (OPE).

5. **Expected Deliverables**  
1. **Written Project Description (via Devpost)**  
* Provide a clear written description of your project that includes:  
  * How your solution addresses the problem statement  
  * Development tools used (e.g. VSCode, Colab, Jupyter)  
  * APIs used (e.g. OpenAI GPT-4o, Google Maps API)  
  * Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)  
  * Datasets and assets used (e.g. Google Local Reviews dataset, manually labelled data)  
2. **Public Code/GitHub Repository**  
* Submit a link to a public Code/GitHub repository containing:  
  * Well-structured, commented code covering all components of your solution  
  * A README file that includes:  
    * Project overview  
    * Setup and installation instructions  
    * Steps to reproduce your results  
    * A brief reflection on your solution's limitations and what you would improve given more time  
    * Team member contributions (if applicable, i.e. team participants, non-solo participants)  
3. **Run & Iteration Logs**  
* Submit the per-iteration log required in the Starter Kit (Run-log requirements), covering:  
  * Hypothesis for that iteration — what the agent intended to try and why  
  * The code diff applied  
  * The resulting metrics (NDCG@10 / Recall@50 for the KuaiRand benchmarks)  
  * Any error or recovery events encountered, and how the agent handled them  
* A short summary reporting the number of manual interventions during the run (used to assess autonomy per Task Requirement 2\)  
4. **Final Submission & Results Summary**  
* Submit your final model output/checkpoint for the required benchmark (KuaiRand-Pure), in the schema defined by the Starter Kit. If you also attempt the bonus benchmarks (KuaiRand-1k & KuaiRand-27k), submit their outputs as well for bonus scoring.  
* A results table reporting your validation-best score for the required benchmark's metrics (KuaiRand-Pure NDCG@10 / Recall@50), and its absolute delta over the official baseline (per the Evaluation section scoring formula); if you attempted the bonus benchmarks (KuaiRand-1k & KuaiRand-27k), include their NDCG@10 / Recall@50 results as well  
* Reported resource usage required to reach the converged result: total token consumption (input \+ output) from the agent's LLM calls, and total GPU time (GPU-hours) consumed during training and evaluation (used to score Feasibility & Practicality)  
6. **Judging Criteria**

| Judging Criteria | Weight |
| :---- | :---: |
| **Technical Execution** | **35%** |
| **Innovation & Problem Insight** | **20%** |
| **Impact & Relevance** | **20%** |
| **Feasibility & Practicality** | **15%** |
| **Presentation & Communication** Final Event Only | **10%** |

### **Technical Execution — Primary Metric & Robustness**

**Primary metric.** We score the **converged result**, not the peak and not the intermediate trajectory. A run is considered converged when **validation score has not improved by more than a small threshold ε over the last *N* consecutive iterations** (default: ε and *N* fixed by the organizers and published in the Starter Kit), *or* when the run hits the fixed compute/wall-clock budget — whichever comes first. The submission scored for ranking is the **validation-best checkpoint** at that point, evaluated **once on the hidden test set**. The agent develops only on train \+ validation; it never sees the hidden test set.

* **KuaiRand-Pure is the required benchmark** and determines 100% of the Primary metric score. **KuaiRand-1k and KuaiRand-27k are bonus benchmarks**: a strong result on either earns additional bonus points on top of the Primary metric score, but skipping them does not reduce the KuaiRand-Pure score.  
* Per-dataset metrics: **KuaiRand-Pure / KuaiRand-1k / KuaiRand-27k** → NDCG@10 / Recall@50. Within each dataset, the score is the **equal-weighted average of each metric's *absolute* improvement over the official baseline** on the hidden test set. For every metric *m*:

delta(m) \= score\_agent(m) − score\_baseline(m)

* score\_dataset \= mean over m of  delta(m)

**Robustness.** Not judged by whether the agent ever hits a failure, but by **how it handles one** — recovering, retrying, or routing around a failed step (a code error, a timeout, an unexpected input) so that long iterative runs neither crash, stall, nor diverge before hitting the compute/wall-clock budget.

### **Innovation & Problem Insight**

Judged on what the agent identified as worth trying and why — not on implementation.

* What the agent chose to target across the full algorithmic stack (features, model architecture, training strategy, evaluation loop, etc. — improvements are not limited to the model itself) and the reasoning behind that choice.  
* Originality in drawing on published methods, papers, or public solutions — rewarding agents that go beyond naive baseline tweaks.

### **Impact & Relevance — Autonomy**

**Autonomy.** How much of the improvement loop the agent drives on its own — proposing and testing changes based on its own evaluation of results, not just tuning the model architecture. Measured primarily by the **number of manual interventions** required to reach the converged result; fewer interventions score higher, with fully autonomous runs scoring highest. The fewer humans required, the more this reflects real acceleration of recommender-system R\&D.

### **Feasibility & Practicality — Resource Consumption**

How much it costs — in both LLM usage and GPU compute time — to reach the converged result.

* **Token consumption.** Total input \+ output tokens used by the agent's LLM calls across the run.  
* **GPU time.** Total GPU-hours consumed during training and evaluation to reach the converged result — captures the actual compute resources used in a way that wall-clock time alone cannot (e.g. running on more GPUs in parallel looks fast on the clock but is not necessarily cheaper).  
7. **References**

\[1\] J. S. Chan, N. Chowdhury, O. Jaffe, J. Aung, D. Sherburn, E. Mays, G. Starace, K. Liu, L. Maksin, T. Patwardhan, L. Weng, and A. Mądry, "MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering," OpenAI, 2024\. arXiv:2410.07095. [https://doi.org/10.48550/arXiv.2410.07095](https://doi.org/10.48550/arXiv.2410.07095)  
\[2\] Z. Jiang, D. Schmidt, D. Srikanth, D. Xu, I. Kaplan, D. Jacenko, and Y. Wu, "AIDE: AI-Driven Exploration in the Space of Code," 2025\. arXiv:2502.13138. [https://doi.org/10.48550/arXiv.2502.13138](https://doi.org/10.48550/arXiv.2502.13138)  
\[3\] Y. Yamada, R. T. Lange, C. Lu, S. Hu, C. Lu, J. Foerster, J. Clune, and D. Ha, "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search," 2025\. arXiv:2504.08066. [https://doi.org/10.48550/arXiv.2504.08066](https://doi.org/10.48550/arXiv.2504.08066)

8. **Appendix A. A Primer on Recommender Systems**

> This appendix gives participants without a recommender-systems background just enough to get started. It is a concept map plus an annotated reading list — not a textbook. Use it to understand the KuaiRand benchmarks and to know what to look up when you get stuck.

### **A.1 The Big Picture: The Recommendation Pipeline**

A modern industrial recommender does not score every item directly. It runs a funnel of stages, each narrowing the candidate set:  
Recall  →  Pre-ranking  →  Ranking  →  Re-ranking  
millions    thousands       hundreds     final list

* **Recall / Retrieval**: cheaply retrieve a few thousand candidates from millions.  
* **Pre-ranking**: a lightweight model trims the candidates further.  
* **Ranking**: a heavy, accurate model scores each candidate. **This challenge mostly lives here.**  
* **Reranking**: adjust the final ordering for diversity, business rules, and so on.

> For this competition you mainly need the **ranking** stage. The KuaiRand benchmarks are ranking/prediction tasks, not full end-to-end pipelines.  
This content is only supported in a Feishu Docs

### **A.2 Core Tasks: CTR and the Feedback Funnel**

Most industrial ranking is framed as predicting the probability of user feedback:

* **CTR (Click-Through Rate)** — `P(click | impression)`. The user saw the item; will they click?  
* **CVR (Conversion Rate)** — `P(conversion | click)`. The user clicked; will they convert (buy)? E-commerce background only; not a task in this challenge.  
* **The funnel**: `impression → click → deeper engagement` (in e-commerce, `→ conversion`). Because these stages are linked, two well-known problems arise:  
  * **Sample selection bias**: the post-click signal is only observed on *clicked* items, yet must be predicted for *all* impressions.  
  * **Data sparsity**: post-click signals such as `long_view` or `like` are far rarer than clicks.

> **KuaiRand** has no purchase label, so CVR itself is never scored here. But the same two problems reappear on its post-click signals (`long_view`, `like`, `follow` …), and ESMM-style multi-task modelling — see A.3 — is a legitimate approach to them.

### **A.3 Multi-Task & Multi-Feedback Learning**

Real users produce many signals (click, like, follow, comment, watch-time, and so on). Predicting them jointly — rather than training a separate model per signal — shares representations and tends to improve every task.

* Why it matters here: **KuaiRand** provides **12 feedback signals**, so a multi-task model can learn from several of them jointly even though only `click` is scored.  
* The key idea is to balance *shared* parameters (which transfer useful knowledge across tasks) against *task-specific* parameters (which prevent conflicting tasks from hurting one another — the "seesaw" problem).

### **A.4 Evaluation Metrics**

| Metric | Intuition | Used for |
| :---- | :---- | :---- |
| **AUC** | Probability that a random positive is ranked above a random negative. Threshold-free and robust to class imbalance. | CTR / CVR prediction in general (not scored in this challenge) |
| **NDCG** | Quality of a *ranked list*, rewarding relevant items near the top (with a position discount). | Ranking quality (KuaiRand) |
| **Recall** | Fraction of all relevant items that appear in the returned list. | Coverage (KuaiRand) |

> **Offline vs. online**: a higher offline metric does not always mean better real-world performance (because of distribution shift and feedback loops). This competition is evaluated offline, but it is worth knowing the gap exists.

### **A.5 Feature Engineering Basics**

* **ID features**: user ID, item ID, category ID — high-cardinality discrete features.  
* **Embedding**: map each discrete ID to a learnable dense vector. This is the foundation of all deep recommenders.  
* **Feature crossing**: combine features (e.g. user × category) to capture interactions. Models such as FM and DeepFM automate this.

### **A.6 Annotated Reading List**

\[Hints: If you find reading the following material challenging or find you have missing backgrounds, you can use ChatGPT / Claude / ... to explain it to you.\]  
The goal here is only to understand **how a recommender system is structured** — the recall → ranking → re-ranking pipeline — and where the ranking stage (which this challenge targets) sits within it. You do **not** need to read a whole course; the introductory overview is enough. **Read just one of the following:**

* Google, *Recommendation Systems* (Machine Learning Crash Course), the **Overview** section — `https://developers.google.com/machine-learning/recommendation` A short, official overview of the pipeline. Note: Google calls the ranking stage **"scoring"** — this is the same thing as **ranking**, and it is the part this challenge focuses on.  
* Wang Shusen, *Recommender Systems*, **Chapter 1 (Overview)** — `https://github.com/wangshusen/RecommenderSystem` The most beginner-friendly Chinese resource; the first chapter alone gives the full architecture.

[image1]: <data:image/png;base64,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>