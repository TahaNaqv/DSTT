# Presentation Script & Q&A Prep
**Anomaly Detection in Blockchain Transactions: Graph-Based and Ensemble ML with Explainable AI**
S Muhammad Taha Abbas Naqvi · 26I-8004 · DSTT

Target length: **about 10 minutes** of speaking (≈1,250 words at a relaxed pace), leaving time for questions. Each slide shows a time budget. If you're told you have less time, use the **[SHORT]** version of each slide.

---

## Part 1 — Slide-by-slide script

### Slide 1 · Title  (≈20 s)
> Good morning/afternoon everyone. I'm Taha Abbas Naqvi, roll number 26I-8004, from MS Data Science.
>
> My proposal is titled *"Anomaly Detection in Blockchain Transactions Using Graph-Based and Ensemble Machine Learning with Explainable AI."*
>
> In one sentence: I want to find out how reliably we can catch illicit Bitcoin transactions, and whether graph neural networks really beat classical models when the comparison is done fairly.

---

### Slide 2 · Outline  (≈15 s)
> I'll start with the background and some key terms, then review six papers. From that I'll derive the problem statement, my research questions and objectives, and finally the proposed methodology.

*Don't read the list. Just gesture to it and move on.*

---

### Slide 3 · Introduction  (≈60 s)
> Blockchains like Bitcoin and Ethereum record every transaction on a public ledger that can't be changed. So the data is fully transparent. But the people behind it are pseudonymous: you see addresses, not names.
>
> That combination is attractive for crime. Money moves across borders, there's no bank in the middle, and identities are hidden. So crypto gets used for money laundering, ransomware payments, darknet markets and scams.
>
> *[point to the stat card]* Chainalysis estimated that around **41 billion dollars** reached illicit addresses in 2024 alone.
>
> From a data science point of view, what makes this interesting is that **transactions form a graph**. Every transaction is a node, and money flowing from one to another is an edge. A normal table of features can't see those connections.
>
> And the timing matters: recent papers from 2025 and 2026 disagree on whether graph neural networks actually help here. That open question is what my project addresses.

**[SHORT]** Public ledger + pseudonymity → crime; $40.9B in 2024; transactions form a graph; recent papers disagree on whether GNNs help.

---

### Slide 4 · Key Terms & Context  (≈60 s)
> Four terms I'll use throughout:
>
> **Transaction graph:** nodes are transactions, and edges are the flow of Bitcoin between them.
>
> **Illicit transaction:** one linked to a criminal entity like a laundering service, ransomware or a scam. This is the rare positive class we're trying to detect.
>
> **Class imbalance:** this is the key challenge. Only about **2% of all transactions**, and under **10% of the labelled ones**, are illicit. So a model that says "everything is licit" gets over 90% accuracy while catching nothing. That's why I won't use accuracy as my main metric.
>
> **Ensemble vs. graph models:** Random Forest and XGBoost learn from each transaction's features in a table. GCN and GAT learn by passing information between neighbouring transactions in the graph.
>
> *[point to the stat boxes]* The dataset is the **Elliptic Bitcoin dataset**, the standard benchmark: about 204 thousand transactions, 234 thousand edges and 49 time steps covering roughly two years. Each node has 165 or more features, and 4,545 transactions are labelled illicit.

**[SHORT]** Define the graph and the illicit class; stress 2% / <10% illicit, so accuracy is useless; give the Elliptic numbers.

---

### Slide 5 · Literature Review (1/2)  (≈70 s)
> I reviewed six papers. The table compares their objective, method, dataset, results and limitations, and how each one is relevant to my work.
>
> **Weber et al., 2019** created the Elliptic dataset. Their surprising result was that a plain **Random Forest beat the GCN**: about 0.67 recall on the illicit class versus 0.51. That was the first sign that graph structure alone doesn't guarantee better results. I'll reproduce their baselines.
>
> **Asiri and Somasundaram, 2025** reported the GCN as best, but they relied mainly on **accuracy**. With a 90:10 class ratio, accuracy hides whether you're actually catching fraud. They also didn't test on truly unseen data. That's exactly why I avoid accuracy as a headline metric.
>
> **Taher et al., 2024** worked on Ethereum and found **XGBoost with explainable AI** gave about 99.6% accuracy. But the dataset is small, under ten thousand transactions, so that number is probably optimistic. It supports using XGBoost plus explanations in my ensemble track.

**[SHORT]** Weber: RF > GCN. Asiri: accuracy-only, misleading. Taher: XGBoost + XAI works, but on small data.

---

### Slide 6 · Literature Review (2/2)  (≈80 s)
> **Chithanuru and Ramaiah, 2025** stacked Random Forest, XGBoost and a neural network, used **SMOTEENN** for imbalance and **SHAP and LIME** for explanations. Again the accuracy is very high on a small Ethereum dataset. I take my imbalance and SHAP choices from this paper.
>
> **Zheng et al., 2025** is the strongest case *for* graph models. Their temporal graph attention network reached a **ROC-AUC of 0.913**, versus 0.836 for XGBoost. But notice that **Random Forest still had the best F1** at a fixed threshold. So "which model is best" depends on the metric you choose.
>
> **Maganti, 2026** is the most important one for me. It re-ran the comparison with **no leakage**, meaning the model never sees the test period's graph during training. Under that stricter setup, **Random Forest got an F1 of 0.82 and the best GNN only 0.69.** So some of the reported GNN advantage may come from leakage in the evaluation.
>
> *[point to the takeaway box]* The takeaway: the literature is split. GNNs look better on AUC or with leaky evaluation, and ensembles look better on thresholded F1. Many studies report accuracy on imbalanced data. So what's missing is a **fair, imbalance-aware, leakage-controlled comparison, with explanations**.

**[SHORT]** Chithanuru → SMOTEENN + SHAP. Zheng → GAT wins AUC, RF still wins F1. Maganti → without leakage, RF 0.82 > GNN 0.69. The literature is split, and a fair comparison is needed.

---

### Slide 7 · Problem Statement  (≈45 s)
> So here is the problem.
>
> *[read the statement slowly, or paraphrase:]* Illicit transactions are a tiny, around 2%, but harmful share of blockchain activity. The severe imbalance and the graph structure leave compliance teams without models that are both **reliable and interpretable**. And we still can't say whether graph neural networks truly beat classical ensembles, because most studies report accuracy on imbalanced data or leak future graph information into training.
>
> *[point to the three boxes]* The **use case** is transaction monitoring for exchanges, regulators and anti-money-laundering analysts. The **gap** is the conflicting, leakage-prone results. The **need** is detection that works under imbalance and that a compliance officer can understand.

---

### Slide 8 · Research Questions  (≈50 s)
> From this problem I derived four research questions.
>
> **RQ1:** once evaluation and imbalance are controlled for, does a graph model like GCN or GAT meaningfully outperform a well-tuned Random Forest or XGBoost? This is the main question.
>
> **RQ2:** which imbalance strategy, among class weighting, SMOTE or SMOTEENN, and threshold tuning, gives the best precision-recall trade-off?
>
> **RQ3:** which features drive the predictions, and do the two model families agree on them?
>
> **RQ4:** a practical one: what does each approach cost to compute and build within a course timeline on free hardware?

---

### Slide 9 · Research Objectives  (≈50 s)
> Each objective maps onto a question. *[point to the arrows on the right]*
>
> **RO1:** characterize the dataset through EDA, meaning the imbalance, the feature distributions and the graph structure. This supports every question.
>
> **RO2:** develop and tune Random Forest, XGBoost and one GNN on the **same** leakage-controlled temporal split, for RQ1.
>
> **RO3:** compare the imbalance strategies using illicit-class F1 and PR-AUC, for RQ2.
>
> **RO4:** evaluate both families on precision, recall, F1, ROC-AUC, PR-AUC and a cost-weighted metric, also for RQ1.
>
> **RO5:** explain predictions with SHAP and graph attribution and measure whether the feature rankings agree, for RQ3.
>
> **RO6:** measure training time and hardware needs and summarize the trade-offs in a dashboard, for RQ4.

---

### Slide 10 · Proposed Methodology  (≈90 s)
> This is my preliminary pipeline. *[trace left to right with your hand]*
>
> **1:** load the Elliptic graph. **2:** preprocess and run EDA: scale the features, build the graph, and, importantly, split **by time**. I train on earlier time steps and test on later ones, the way Weber did with steps 1 to 34 for training and 35 to 49 for testing. Transactions never mix across that line.
>
> **3:** handle the imbalance with class weights, SMOTE or SMOTEENN, and threshold tuning. Oversampling is applied **only to the training data**.
>
> Then the pipeline splits into two tracks that run on the **same split**. **4a** is the ensemble track with Random Forest and XGBoost. **4b** is the graph track with a GCN or GAT in PyTorch Geometric.
>
> Both feed into the same **5, evaluation**, then **6, explainability** (SHAP for the tree models and attention or attribution for the GNN), and **7, a comparative report and dashboard**.
>
> *[point to the bottom-left card]* These choices come from the base papers. The baselines come from Weber, Taher and Chithanuru, the GNNs from Weber and Zheng, and the leakage-free evaluation from Maganti.
>
> *[bottom-right card]* On feasibility, I'm being careful not to overclaim. The ensemble track trains in minutes on a laptop. For the graph track, I'll limit myself to **one** well-tested GNN on free Colab or Kaggle GPUs. The exact hyperparameters and final architecture will be decided during implementation, and the Ethereum extension only happens if time allows.

---

### Slides 11–12 · References  (≈10 s total)
> These are my 14 references in IEEE format. They cover the six reviewed papers, the dataset, and the standard sources for Random Forest, XGBoost, GCN, GAT, SMOTE and SHAP.

*Click through quickly; don't read them.*

---

### Slide 13 · Thank You  (≈20 s)
> To sum up: this project runs a **fair, leakage-controlled comparison** of classical ensembles and graph neural networks for illicit Bitcoin transaction detection, uses **imbalance-aware metrics**, and **explains** the predictions so a compliance analyst could use them.
>
> Thank you. I'm happy to take questions.

---

## Part 2 — Cheat sheet: numbers to know by heart

| Item | Value |
|---|---|
| Elliptic nodes / edges | 203,769 transactions / 234,355 directed edges |
| Time steps | 49 (each ≈ 2 weeks; ≈ 2 years in total) |
| Features | 166 in the original (94 local incl. time step + 72 aggregated neighbour features); 183 in Elliptic++ |
| Labelled nodes | 46,564 (≈ 23%): **4,545 illicit (9.8%)**, 42,019 licit |
| Unlabelled ("unknown") | ≈ 157,000 (≈ 77%) |
| Illicit share of *all* nodes | ≈ 2.2% |
| Standard split | Train time steps 1–34, test 35–49 (Weber et al.) |
| Weber 2019 | RF illicit recall ≈ 0.67 vs GCN ≈ 0.51 |
| Zheng 2025 | ATGAT AUC 0.913 vs XGBoost 0.836 vs GCN 0.814; RF best F1 |
| Maganti 2026 | Inductive, leakage-free: RF F1 0.821 vs best GNN F1 0.689 |
| Ethereum papers | ≈ 9,840 transactions, ≈ 99.6% accuracy |
| Chainalysis | ≈ $40.9B to illicit addresses in 2024 |

**30-second summary (if asked "so what is your project?"):**
> I'm comparing classical ensemble models and graph neural networks for detecting illicit Bitcoin transactions on the Elliptic dataset. Previous papers disagree on which is better, partly because they use accuracy on very imbalanced data or evaluation that leaks test information. I'll use a strict time-based split, imbalance-aware metrics like PR-AUC and F1, and SHAP-style explanations to give a fair, interpretable answer.

---

## Part 3 — Likely questions and how to answer them

### A. Motivation and novelty
**Q: What's novel here? Aren't you just re-running existing models?**
> The contribution isn't a new algorithm. It's a **careful, fair comparison**. The literature disagrees because of how models are evaluated, so the value is in running both model families on the **same leakage-free temporal split**, with **imbalance-aware metrics**, comparing **several imbalance strategies side by side**, and checking whether their **explanations agree**. As far as I found, most papers do only one or two of these. For a course project, doing the evaluation properly is itself the contribution.

**Q: Why blockchain? Why not credit-card fraud?**
> The course covers AI, data science and blockchain, and blockchain data is naturally a graph. That lets me compare graph and non-graph models, which you can't really do with tabular credit-card data. It's also a real, current problem for exchanges and regulators.

### B. Dataset
**Q: Why Elliptic and not a newer or bigger dataset?**
> It's the standard benchmark, so my results can be compared directly to Weber, Zheng and Maganti. It's labelled, public, and small enough to run on free hardware. Elliptic++ is a possible extension.

**Q: What do the features mean?**
> The first 94 are local: the time step plus information about the transaction itself, such as number of inputs and outputs, fees and amounts. The other 72 aggregate information from the transaction's direct neighbours. **Important caveat:** Elliptic anonymized the feature names for privacy, so explanations will say things like "local feature 53" rather than "transaction fee". I can still show whether local or neighbourhood features matter more, and whether the models agree.

**Q: What will you do with the 77% unknown nodes?**
> For the ensembles I only train and test on labelled nodes. For the GNN, unknown nodes **stay in the graph** so messages can flow through them, but they aren't counted in the loss or the metrics. That's the standard semi-supervised setup.

**Q: How reliable are the labels?**
> They come from Elliptic's investigations and heuristics, so there may be noise or bias. That's why I present outputs as **decision support for an analyst, not proof** of wrongdoing. It's in my ethics section.

**Q: Bitcoin vs Ethereum, is there a difference?**
> Yes. Bitcoin uses a UTXO model where transactions link to each other, while Ethereum is account-based, so its graph is between wallets. That's why the Ethereum set is only an optional generalization check, not the main experiment.

### C. Methodology and leakage
**Q: What exactly is "leakage" / transductive vs inductive?**
> **Transductive:** the model sees the whole graph during training, including the structure and features of test-period nodes, and only their labels are hidden. **Inductive:** test-period nodes and edges are completely unseen until test time. In the real world you can't see future transactions, so inductive is the realistic setting. Maganti showed GNNs lose much of their advantage under it.

**Q: How will you do the split?**
> By time: train on the earlier time steps (e.g., 1–34) and test on later ones (35–49), with a validation block taken from the end of the training period (e.g., steps 30–34) for tuning. No random shuffling, because random splits let the model learn from the future.

**Q: Why might a Random Forest beat a GNN on a graph problem?**
> Three reasons. First, the 72 aggregated features already encode the 1-hop neighbourhood, so the tabular model already "sees" part of the graph. Second, the graph is very sparse, with about 1.15 edges per node, so message passing adds little. Third, there's strong **temporal drift**: GNNs can overfit to the structure of the training period.

**Q: Can you apply SMOTE to a graph?**
> Not directly: synthetic nodes have no real edges. So I'll use **SMOTE/SMOTEENN only in the ensemble track**, and **class-weighted loss plus threshold tuning** for the GNN. Class weighting and threshold tuning are used in both tracks so the comparison stays fair. SMOTE is applied only to training data, never to validation or test.

**Q: Why GCN and GAT and not GraphSAGE or EvolveGCN?**
> GCN is the classic baseline from Weber, and GAT's attention weights give some interpretability, which fits my explainability goal. GraphSAGE is naturally inductive and would be a good stretch goal. EvolveGCN is heavier than I can justify in the time available. I'm scoping to one well-tuned GNN rather than many half-tuned ones.

**Q: How will you tune hyperparameters fairly?**
> Same budget for both families: randomized or grid search on the temporal validation block, selecting by PR-AUC or F1. I'll also run several random seeds and report the mean and standard deviation, so a difference isn't just luck.

**Q: What about the dark-market shutdown?**
> Weber et al. found that after a large dark market was shut down around time step 43, **all** models stopped detecting new illicit patterns. I'll report per-time-step performance to show that drift, not just one averaged number.

### D. Evaluation
**Q: Why PR-AUC over ROC-AUC?**
> ROC-AUC uses the false-positive rate, which stays tiny when there are many licit transactions, so it can look good even with many false alarms. PR-AUC focuses on the illicit class. Its random baseline is the illicit prevalence (about 10%), not 0.5, so it shows real skill under imbalance.

**Q: What is your cost-based metric?**
> A weighted error: total cost = c_FN × missed illicit + c_FP × false alarms, with a missed illicit transaction weighted higher, e.g., 5–10×. The exact ratio is a policy choice, so I'll show how the best threshold changes as the ratio changes rather than claim one "true" cost.

**Q: How will you decide which model "wins"?**
> Mainly by illicit-class F1 and PR-AUC on the held-out time steps, averaged over seeds. If the difference is within the seed variance, I'll call it a tie. Then I'll weigh it against interpretability and compute cost (RQ4).

**Q: What if the GNN loses?**
> That's still a valid, useful result. It would confirm Maganti's finding under my setup and tell practitioners that a cheaper, more interpretable model is enough. The project is designed so any outcome answers RQ1.

### E. Explainability
**Q: How do you explain a GNN?**
> GAT attention weights show which neighbours the model focused on, and tools like GNNExplainer or gradient-based attribution give feature importance. One caveat: attention weights aren't guaranteed to be faithful explanations, so I'll treat them as indicative.

**Q: How will you check whether the two families "agree"?**
> Compare the top-k features from SHAP with the GNN's attribution ranking, using the top-k overlap and Spearman rank correlation.

**Q: Why SHAP?**
> It has a solid game-theory basis. TreeSHAP is exact and fast for Random Forest and XGBoost. It also gives both global importance and per-transaction explanations, which is what an analyst needs.

### F. Feasibility, scope, ethics
**Q: Can you do all of this in one semester?**
> The ensemble track is low-risk: it trains in minutes. The GNN is limited to one architecture, and the Elliptic graph fits comfortably in a free Colab GPU. The Ethereum extension and the dashboard polish are optional. If time runs short, the core deliverable, RQ1 to RQ3 on Elliptic, is still complete.

**Q: Could this be used in production?**
> Not directly. It's a methodological study. But the findings (which model, which threshold, which features) would inform a real monitoring system where flagged transactions go to a human analyst.

**Q: Ethical concerns?**
> The data is public and already pseudonymized, with no personal information. Labels may be biased, so outputs are decision support, not accusations. I won't make claims about any real entity.

### G. Questions about your sources (be ready)
**Q: The Maganti paper is from 2026. Is it peer-reviewed?**
> It's an arXiv preprint, so not yet peer-reviewed. That's why I use it as **motivation for a careful protocol**, not as settled fact. My project effectively tests its claim.

**Q: Where does the $40.9B figure come from?**
> The Chainalysis 2025 Crypto Crime Report, which I cite as quoted in Zheng et al. [6]. *(Before presenting, check the original report yourself. Chainalysis sometimes revises its estimates upward later.)*

---

## Part 4 — Delivery tips
- **Rehearse aloud twice with a timer.** Most people run 30–50% over on the first try.
- **Slides 5, 6 and 10 are where you earn marks.** Know them well enough to talk without reading.
- **Don't read the table.** Name the paper, give one result and one limitation, and say why it matters to you.
- **On the methodology slide, trace the diagram with your hand** from left to right.
- **If you don't know an answer:** "That's a good point. I haven't settled that yet; my current plan is X, and I'll confirm it during implementation." This matches the template's "don't overclaim" guidance.
- **Repeat the key message** at the start (slide 1) and the end (slide 13): *fair, leakage-controlled, imbalance-aware, explainable.*
- **Presenter View:** this script is also in each slide's speaker notes in the .pptx.
