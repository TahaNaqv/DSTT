# Part 5 — Lectures 7 & 8: EDA, Visualization, Data Storage, Intro to Machine Learning

---

## 5.1 Why find patterns? The five framing questions (Lec 7 opens with these)
1. Why is it important to find patterns?
2. What counts as a pattern?
3. What techniques can we use to find patterns?
4. When can such techniques be used?
5. How should the results be interpreted?

## 5.2 The Data Analysis Template
| Component | Purpose |
|---|---|
| **Exploratory Data Analysis** | **Summary of the data**; **accidental and unexpected patterns** |
| **Data Screening** | **Check for statistical glitches** |

## 5.3 The role of EDA
> Explore a data set — **use methods that help you understand the data**:
> - to help you **understand the events that generated the data**
> - to help you **see what happened, sometimes in spite of your expectations**

> **EDA supplies statistical techniques that work in combination with a very powerful pattern
> recognition device — the human eye/brain.** It gives you ways to **tabulate, summarise, display,
> reduce** data. *"A representation that the eye finds congenial is of more practical use"* than a
> table of raw data.

## 5.4 Data Analysis can't be done mechanically
- Often there has to be a **"creative" element**.
- **Conventional DA is in a sense idealistic.**
- There's a trade-off between **"ideal" experimentation vs ecological validity**.
- Sometimes questions are **tentative**.
- *"We need data analysis skills that allow data to speak to us **despite our expectation**."*

---

## 5.5 ★★ CONFIRMATORY vs EXPLORATORY DATA ANALYSIS (near-guaranteed exam question)

| | **Exploratory Data Analysis (EDA)** | **Confirmatory Data Analysis (CDA)** |
|---|---|---|
| Purpose | **Finds a good description** | **Tests a hypothesis** |
| Effect on questions | **Raises new questions** | **Settles questions** |
| Statistics branch | **Descriptive statistics** | **Inferential statistics** |
| Attitude | Data-driven, open-ended, hypothesis-*generating* | Theory-driven, hypothesis-*testing* |
| Typical output | Plots, summaries, candidate patterns | p-values, confidence intervals, accept/reject |

**Father of EDA: John Tukey**, *Exploratory Data Analysis*, Addison-Wesley, **1977** — the
foundational text that coined the term. (Also cited: Wickham & Grolemund, *R for Data Science*, 2017, ch. 7.)

---

## 5.6 Simple example — class attendance and language learning

| Student | Classes | Words |
|---|---|---|
| Bob | 10 | 100 |
| Carol | 15 | 150 |
| Dave | 12 | 120 |
| Ann | 17 | 170 |
| **Steve** | **13** | **95** |

**The pattern:** for Bob, Carol, Dave and Ann, **words = 10 × classes** exactly. **Steve breaks the
pattern** — 13 classes should give 130 words, but he learned 95.
**That is the whole point of EDA:** the summary statistic (a correlation, a mean) would hide Steve;
*looking* at the data exposes him as the interesting case worth explaining (an outlier / a new
hypothesis: illness? different method? data error?).

---

## 5.7 What is data? Data reduction

> *A bunch of numbers (usually). **Each number summarises some property or event of interest.***
> e.g. **18** could be an Age, a **Beck Depression Inventory (BDI) score**, or Income in £'000s.
> Data: lots of numbers — 18, 24, 43, 22, 37, … **Is there a pattern?**

**Data reduction = fewer numbers.**

| Type | Raw | Re-presented |
|---|---|---|
| **Summarise proportion** | 27/48 children in class A are boys; 16/23 in class B | **56% of class A, 69% of class B are boys** |
| **Summarise change** | Before: 112, 134, 121, 97 → After: 116, 132, 140, 108 | **Change: +4, −2, +19, +11** |

> Reducing to **percentages** makes unequal group sizes comparable; reducing to **change scores**
> makes the effect visible without the baseline noise. Both are EDA.

---

## 5.8 Revealing patterns — the nitrogen-weight table
> **Raw data is hard to understand. EDA provides ways of presenting data that make it easier to understand.**

The slide shows Lord Rayleigh's classic table (Date | Source compound | Extraction method | Weight
observed): weights from **NO / N₂O / NH₄NO₂** cluster around **2.2989–2.3018**, while those from
**Air** cluster around **2.3101–2.3103**. You cannot see that in the table; the moment you draw a
**dot plot or two side-by-side box & whisker plots**, the two groups separate cleanly. *(Historically
this gap led to the discovery of argon.)*

**Plots used on these slides:** **box & whisker plot**, **dot plot**, **two separate box & whisker plots**.

**Technique slide:** *"Find a graph that shows clearly that the data can be divided into different
groups. **Appropriate representation depends on your practical goal.**"*

> **Multiplicity** (Interpreting EDA slide): if you look at enough plots/subgroups, some
> "patterns" will appear by chance alone. This is the multiple-comparisons problem, and it's why
> EDA findings are hypotheses to be confirmed later, not conclusions.

---

## 5.9 The class activity table (Lec 7 slide 16) — be ready to answer these

| ID | Age | Sex | Education | Marital-Status | Hours/Week | Income |
|---|---|---|---|---|---|---|
| 1 | 25 | Male | Bachelors | Never-married | 40 | ≤50K |
| 2 | 38 | Male | HS-grad | Married-civ-spouse | 50 | >50K |
| 3 | 28 | Female | Masters | Never-married | 45 | >50K |
| 4 | 52 | Male | HS-grad | Married-civ-spouse | 60 | >50K |
| 5 | 45 | Female | Bachelors | Divorced | 30 | ≤50K |
| 6 | 36 | Male | Some-college | Married-civ-spouse | 55 | >50K |
| 7 | 61 | Male | HS-grad | Married-civ-spouse | 20 | ≤50K |
| 8 | 29 | Female | Assoc-acdm | Never-married | 40 | ≤50K |
| 9 | 34 | Male | Bachelors | Married-civ-spouse | 70 | >50K |
| 10 | 41 | Female | HS-grad | Married-civ-spouse | 45 | >50K |
| 11 | 23 | Male | HS-grad | Never-married | 15 | ≤50K |
| 12 | 55 | Female | Doctorate | Married-civ-spouse | 60 | >50K |

**The instructor's own questions (from the speaker notes) and their answers:**

1. **How many males vs females?** → **7 male, 5 female.**
2. **Most common education level?** → **HS-grad (5 of 12)**; then Bachelors (3); Masters, Some-college, Assoc-acdm, Doctorate 1 each.
3. **Which income group works longer hours on average?**
   - **>50K** (IDs 2,3,4,6,9,10,12): 50+45+60+55+70+45+60 = **385 / 7 = 55.0 hrs**
   - **≤50K** (IDs 1,5,7,8,11): 40+30+20+40+15 = **145 / 5 = 29.0 hrs**
   - → **The >50K group works far longer hours (55 vs 29).**
4. **Inconsistencies or missing values, and how to fix?** → In this clean extract there are none;
   in the **full Adult dataset**, missing values appear as the string **`?`** in `workclass`,
   `occupation`, `native-country`, plus **leading spaces** on every categorical value
   (`" Male"`, `" Bachelors"`). Fix: `df.replace('?', np.nan)` and `.str.strip()`, then impute
   with the **mode** or an explicit `"Unknown"` category.
5. **Relationship between education and income?** → Higher education is associated with >50K
   (**Masters and Doctorate are both >50K**), but the relationship is **not deterministic**: HS-grad
   appears in **both** classes (IDs 2, 4, 10 are >50K; IDs 7, 11 are ≤50K). So **hours-per-week and
   marital status also matter** — see (6).
6. **Which 2 features are most predictive of income?** → **Hours-per-week** (55 vs 29 average — the
   sharpest separation) and **Marital-Status** (**all 7 `Married-civ-spouse` except ID 7 are >50K;
   all 4 `Never-married` are ≤50K**). Education comes third because of the HS-grad overlap.

> **Age is a poor single predictor here** — both classes span roughly the same range (23–61 vs 25–55).
> Saying so shows you actually read the table.

---

## 5.10 Visualization: the Don'ts and the Do's

**Charts covered:** Bars vs Lines (Zacks 1999) · Yahoo! Finance charts · **Pie charts** ·
**Stacked area chart** (S. Few) · Correlations · **Scatterplots** (xkcd 388) · 3D plots (matplot3d) ·
Distributions · **Histogram** (ggplot2) · **Density plots** · **2D density plots** · Seaborn.
Reference given: **"Choosing a good chart"** (extremepresentation.typepad.com) and the **matplotlib gallery**.

**The "Don't!" slides — the standard critique points:**
| Don't | Why |
|---|---|
| **Gratuitous 3D** (3D bars, 3D pies, matplot3d for 2D data) | Perspective distorts area/length judgements; occludes data. **Use 2D.** |
| **Pie charts** (especially many slices / multiple pies) | Humans compare **angles/areas** badly. A **bar chart** is almost always more readable. |
| **Stacked area charts** | Only the **bottom** series has a common baseline; the others are read as thickness, which is very hard. |
| **Truncated / non-zero y-axis on bars** | A **bar encodes length**, so the axis **must start at zero**; a **line** encodes position, so it may be truncated. That's the "**Bars vs Lines**" slide. |
| **Chartjunk, heavy gridlines, rainbow colormaps** | Ink that carries no data; rainbow maps create false boundaries. |

**Rule of thumb:** *the appropriate representation depends on your practical goal* — pick the chart
from the question you're answering, not from the chart menu.

---

## 5.11 ★ EDA techniques by DATA TYPE (Lec 8 table — memorise)

| Type of data | Suggested EDA technique |
|---|---|
| **Categorical** | **Descriptive statistics** (frequency tabulation) |
| **Univariate continuous** | **Line plot, Histograms** |
| **Bivariate continuous** | **2D scatter plots** |
| **2D arrays** | **Heatmap** |
| **Multivariate: trivariate** | **3D scatter plot**, or a 2D scatter plot with a **3rd variable represented in different colour, shape or size** |
| **Multiple groups** | **Side-by-side boxplot** |

## 5.12 ★ EDA techniques by OBJECTIVE (Lec 8 table — memorise)

| Objective | Suggested EDA technique |
|---|---|
| **Getting an idea of the distribution of a variable** | **Histogram** |
| **Finding outliers** | **Histogram, scatterplots, box-and-whisker plots** |
| **Quantify the relationship between two variables** (one exposure, one outcome) | **2D scatter plot +/ curve fitting; covariance and correlation** |
| **Visualize the relationship between two exposure variables and one outcome variable** | **Heatmap** |
| **Visualization of high-dimensional data** | **t-SNE or PCA + 2D/3D scatterplot** |

---

## 5.13 The EDA taxonomy: Graphical vs Non-Graphical × Univariate vs Multivariate

|  | **Univariate** | **Multivariate** |
|---|---|---|
| **Non-graphical** | Tabulation of data; **central tendency (mode, mean, median)**; **spread (variance, standard deviation, interquartile range, maximum and minimum value)** | **Covariance and correlation** |
| **Graphical** | Histogram, box plot, density plot, stem-and-leaf | Scatter plot, heatmap, scatter matrix, side-by-side boxplots |

### Non-graphical univariate EDA — the three pillars
1. **Tabulation of data** — e.g. the slide's table: Green ball 15 (**75%**), Red ball 5 (**25%**), Total 20 (100%).
2. **Central tendency parameters = Mode, Mean, Median**
3. **Spread of data = variance, standard deviation, interquartile range, maximum and minimum value**

### Central tendency and skewness (the two curve pictures)
- **Symmetrical distribution:** **Mode = Mean = Median** — all three coincide at the peak.
- **Asymmetrical (skewed) distribution:** the three separate. **The mode is always at the peak;
  the mean is pulled furthest toward the long tail; the median lies between them.**
  So for a **right-skewed** distribution: **Mode < Median < Mean.**
  For a **left-skewed** distribution: **Mean < Median < Mode.**

> ⚠️ The lecture slide labels the skewed curve **Mode, Mean, Median** left-to-right. The
> statistically standard ordering for a right-skewed curve is **Mode < Median < Mean** (the mean is
> the most affected by the tail). If the question is multiple-choice, the safe, defensible answer is
> **Mode < Median < Mean**; if he asks you to reproduce the slide, note that the peak is the mode
> and the tail pulls the mean. **Consequence to state either way: the median is the robust measure
> of centre for skewed data — which is exactly why we impute with the median.**

### Formulas to have ready
- **Variance** σ² = Σ(xᵢ − μ)² / N · **Std dev** σ = √σ²
- **IQR = Q3 − Q1** · Outlier fence: **below Q1 − 1.5·IQR or above Q3 + 1.5·IQR**
- **Covariance** cov(X,Y) = Σ(xᵢ−x̄)(yᵢ−ȳ)/(n−1) — **unbounded, unit-dependent**
- **Pearson correlation** r = cov(X,Y)/(σₓσᵧ) — **bounded in [−1, +1], unitless**
- **Correlation is standardized covariance.** Both measure only **LINEAR** association, and
  **correlation ≠ causation.**

**Box plot anatomy:** min (or lower fence) — **Q1** — **median** — **Q3** — max (or upper fence),
with points beyond the fences drawn as outliers. Box width = IQR.

---

## 5.14 After EDA: Data Warehousing and Data Lakes

> **Data warehousing is required to support long-term storage, scalability, and efficient processing of data.**

### Why (the instructor's 4 numbered reasons — a likely long-answer question)

**1. Centralized Storage for Large Datasets**
EDA typically works with **subsets of data or exploratory samples**, but organizations deal with
massive amounts of data from various sources.
- **Data Warehouses** provide a **structured, centralized place to store cleaned, historical, and
  structured data for querying.**
- **Data Lakes** offer **flexible storage for raw, semi-structured, and unstructured data** that may
  not be fully explored during EDA but is crucial for future use in data science and ML.

**2. Preservation of Raw and Processed Data**
- **Warehouses** store **processed data in a consistent, structured form**, readily accessed for
  analysis, reporting and dashboards.
- **Lakes** allow storage of **raw data (before processing)** for future explorations, advanced
  analytics, or ML models.

**3. Supporting Future Analysis and Reusability**
Warehousing and lakes provide **scalable storage** allowing **re-analysis, data re-integration, or
merging with external datasets**.
- Warehouses ensure preprocessed data is available for **fast querying in BI applications** or
  operational decision-making.
- Lakes allow analysts/scientists to **revisit raw data, discover new patterns, or run experiments**
  on historical data that wasn't part of the EDA.

**4. Scalability and Performance**
- **Warehouses are optimized for query performance over large, structured datasets.**
- For data needing **more intensive computation or experimentation** (log analysis, IoT data, or
  unstructured data like images and videos), **data lakes provide scalable storage and flexibility.**

### ★ Data Warehouse vs Data Lake — the comparison table

| | **Data Warehouse** | **Data Lake** |
|---|---|---|
| **Data** | Cleaned, **processed**, historical, **structured** | **Raw**, semi-structured and unstructured |
| **Schema** | **Schema-on-WRITE** (structure defined before loading) | **Schema-on-READ** (structure applied when queried) |
| **Users** | Business analysts, BI, reporting, dashboards | Data scientists, ML engineers |
| **Purpose** | **Fast querying**, BI, operational decision-making | Exploration, advanced analytics, ML, future/unknown use |
| **Cost per TB** | Higher | Lower |
| **Agility** | Rigid but reliable | Flexible; risks becoming a **"data swamp"** without governance |

---

## 5.15 DATA ENGINEERING (Lec 8)

> **"Different at different scales…"**

| Layer | Contents |
|---|---|
| **compute** | code, **python, R, julia, spark, hadoop** |
| **storage/database** | **git, SQL, NoSQL, HBase, disk, memory** |
| **devops** | **AWS, docker, mesos, repeatability** |
| **product** | **database, web, API, viz, UI, story** |

### What kind of data STORAGE do you need? (the escalation ladder)
- **memory** — fastest; fine while the data fits in RAM
- **disk** — *what if we do not fit (in memory)?*
- **cluster** — *what if we still do not fit?*
- **cluster** — *what if we need/can use parts? **What if we MUST bring compute to disk?***

> The last line is the **data locality** principle behind Hadoop/MapReduce: when data is too big to
> move, **ship the computation to where the data lives** rather than shipping data to the compute.

### What kind of data ACCESS do you need? (the 4 database families)

| Family | Systems |
|---|---|
| **relational** | **pandas, SQL: Postgres, sqlite, HBase, VoltDB** |
| **document oriented** | **MongoDB, CouchDB** |
| **key-value** | **Riak, Redis, Memcached** |
| **graph oriented** | **Neo4J** |

> Pick by **access pattern**: joins & aggregates → relational; nested/variable documents → document;
> fast lookup by a single key → key-value; relationship traversal (friends-of-friends, fraud rings) → graph.

---

## 5.16 ★ INTRODUCTION TO MACHINE LEARNING (Lec 8, slides 14–35)

### Definitions (both are examinable)
> **"To improve automatically through experience and by the use of data."**
> **"Field of study that gives computers the capability to learn without being explicitly
> programmed." — Arthur Samuel**

**The programming-vs-ML box diagram (a classic exam figure):**

| | Inputs to the computer | Output |
|---|---|---|
| **Traditional Programming** | **Data + Program** | **Output** |
| **Machine Learning** | **Data + Output** | **Program** (the model) |

> *For a given training data, ML can discover the behavior of the system.*

**The AutoML view:** **Dataset + Optimization Metric + Constraints (Time/cost)** → *Automated Machine
Learning* → **Machine Learning Model**.
**The pipeline view:** **Data → Automated ML (Algorithms/Techniques) → Model**, where *"you will
design it"* and *"Outcome: it will be used for predictions."*

### Applications of Machine Learning (the 4-group list)
| Group | Examples |
|---|---|
| **Image Processing, Computer Vision and Robotics** | **Face recognition, handwriting recognition, autonomous cars** |
| **Business** | **Fraud detection, credit approval, product recommendation** |
| **Communication** | **Spam filtering (e-mail), analysis of caller information** |
| **Other** | **Recommender systems, natural language processing, weather forecasting, computer games** |

---

### ★★ SUPERVISED LEARNING

> **Given: Training data AND the corresponding correct output (Labels).**
> Training data: **(x₁, y₁), …, (xₙ, yₙ)** where **xᵢ ∈ ℝᵈ** and **yᵢ is the label**.

The data matrix picture: each **row** is an example xᵢ = (xᵢ₁, xᵢ₂, …, xᵢ_d) with a label yᵢ attached.
Worked example table:

| fruit | length | width | weight | label |
|---|---|---|---|---|
| fruit 1 | 165 | 38 | 172 | **Banana** |
| fruit 2 | 218 | 39 | 230 | **Banana** |
| fruit 3 | 76 | 80 | 145 | **Orange** |
| fruit 4 | 145 | 35 | 150 | **Banana** |
| fruit 5 | 90 | 88 | 160 | **Orange** |

**Flow:** All labeled data → **Model** → Prediction.

#### The two supervised tasks

| | **Classification** | **Regression** |
|---|---|---|
| **Given** | Data | Data |
| **Prediction** | **Class** | **Numerical value (on a continuous scale)** |
| **y is** | **discrete**. To simplify, **y ∈ {−1, +1}** | **a real value, y ∈ ℝ** |
| **The function** | **f : ℝᵈ → {−1, +1}**, called a **binary classifier** | **f : ℝᵈ → ℝ**, called a **regressor** |
| **Examples from slides** | **Approve credit yes/no, spam/ham, banana/orange** | **Amount of credit, weight of fruit; income as a function of age; weight of the fruit as a function of its length** |

**Decision boundary:** in the Feature-1 vs Feature-2 scatter, the curve separating the blue class
from the red class is the **decision boundary**. The classification slide shows five variants:
a **straight line** (linear), a **smooth curve** (nonlinear), **disconnected/irregular regions**,
an **enclosed island**, and a **3-class** boundary.

**Methods for classification (name at least four):**
> **Support Vector Machines, neural networks, decision trees, K-nearest neighbors, naive Bayes, etc.**

**Regression fit progression (three slides in sequence = the bias–variance story):**
1. A **straight green line** — a simple linear fit. May **underfit** (high bias).
2. A **smooth blue curve** — a polynomial fit that follows the trend better. **Good fit.**
3. A **wiggly red curve** passing through every point — **overfitting** (high variance): it models
   the noise, not the signal, and will generalize badly to new data.

> That three-slide sequence is almost certainly worth a question: *"What do the green, blue and red
> curves illustrate?"* → **underfitting, good fit, overfitting.**

---

### ★★ UNSUPERVISED LEARNING

> **Given: Unlabeled raw data, to find patterns / behavior of the system.**
> Training data: **x₁, …, xₙ, xᵢ ∈ X ⊂ ℝⁿ** — **examples x only, no labels y.**

**Flow:** All **unlabeled** data → **Model** → **Clustering.**

**Clustering / segmentation:**
> **f : ℝᵈ → {C₁, … C_k}** (a set of clusters).
> *Example: find clusters in the population, fruits, species.*

The slide pair shows an unlabeled 2-D scatter that visibly contains **two blobs**, then the same
scatter with **two red ellipses drawn around them** — that is what a clustering algorithm outputs.

### ★ Supervised vs Unsupervised — the answer table

| | **Supervised** | **Unsupervised** |
|---|---|---|
| Input | **Labeled** data (x, y) | **Unlabeled** data (x only) |
| Goal | Learn a **mapping** f: x → y | **Discover structure / patterns** |
| Tasks | **Classification, Regression** | **Clustering**, dimensionality reduction, association |
| Output | **Class** or **numeric value** | **Clusters / groups / compressed representation** |
| Evaluation | Compare prediction to the **known** label (accuracy, RMSE) | **No ground truth** — internal measures (silhouette, inertia) |
| Algorithms | SVM, neural nets, decision trees, kNN, naive Bayes, linear/logistic regression | **k-Means, DBSCAN**, hierarchical clustering, **PCA** |

---

### ★ MACHINE LEARNING: OVERALL FLOW (5 boxes — reproduce this diagram)

```
Problem Nature  →  Training Data  →  Model Training      (f(x))
Identification     Collection                │
                                             ▼
   Model        ←   Model Validation   ←─────┘
 Deployment
```

1. **Problem Nature Identification** — is it classification, regression or clustering?
2. **Training Data Collection**
3. **Model Training** — learn **f(x)**
4. **Model Validation**
5. **Model Deployment**

> Map it onto the DS life cycle from Lecture 2: stages 1 → 2 → 7 → 8 → 9. If he asks how ML fits
> into data science, say: **ML is stage 7 (modeling) of the data science life cycle — one step, not
> the whole thing.**

---

## 5.17 Interpreting EDA — how to close an EDA answer (the final slide)

**After EDA you should:**
- **Summarise the results**
- **Discover unanticipated results** → a new line of research, a new experiment; qualify the
  conclusion from the present study
- **Generate hypotheses**
- **Check assumptions** → qualify the conclusion from the present study; address anomalies

> **And critically: EDA is NOT (or only rarely) a definitive conclusion.**
> That single sentence is the best closing line for any EDA essay question.
