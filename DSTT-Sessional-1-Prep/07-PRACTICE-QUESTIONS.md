# Practice Questions with Model Answers — Sessional-I

> Cover the answers. Write yours first. Then compare.

---

## SECTION A — MCQ / one-liners (20)

1. Temperature in °C is which level of measurement? → **Interval** (no true zero).
2. Temperature in Kelvin? → **Ratio** (inherent zero).
3. Zip code is numeric. What type is it? → **Nominal** (qualitative) — arithmetic on it is meaningless.
4. Size = {small, medium, large} → **Ordinal**.
5. A medical test result (positive/negative) is which kind of binary? → **Asymmetric binary**; assign **1** to the more important outcome.
6. Binary attributes are a special case of ___ attributes. → **Discrete**.
7. "Fewer than 1% of values are non-zero" describes which characteristic? → **Sparsity** (and it is an *advantage*).
8. "The surface of the Earth seems uneven at metres but smooth at tens of kilometres" illustrates → **Resolution**.
9. Which preprocessing technique makes data "more stable" / less variable? → **Aggregation**.
10. Sampling in which the same object can be picked more than once → **Simple random sampling with replacement**.
11. Splitting data into partitions and drawing random samples from each → **Stratified sampling**.
12. Purchase price and sales tax paid are ___ features. → **Redundant**.
13. Student ID for predicting GPA is an ___ feature. → **Irrelevant**.
14. Feature selection that occurs naturally as part of the algorithm → **Embedded**.
15. Feature selection that uses the mining algorithm as a black box → **Wrapper**.
16. Which dimensionality reduction method is supervised? → **LDA**.
17. Which produces embeddings where inter-cluster distances are NOT meaningful? → **t-SNE**.
18. In pandas, `pd.read_excel(file, sheet_name=None)` returns → **a dict of ALL sheets {name: DataFrame}**.
19. In matplotlib, `'k'` is which colour? → **black**.
20. Machine Learning takes ___ and ___ as input and produces ___. → **Data + Output → Program (model)**.

---

## SECTION B — Short answers

**Q1. Define Big Data and list its dimensions.**
> Big Data is **any data that is expensive to manage and hard to extract value from**. Dimensions:
> **Volume** (size of the data), **Velocity** (the latency of data processing relative to the
> growing demand for interactivity), and **Variety & Complexity** (diversity of sources, formats,
> quality, structures). Two further dimensions taught: **Veracity** (trustworthiness) and **Value**.
> Note the definition is economic, not a fixed size threshold.

**Q2. Differentiate structured, semi-structured and unstructured data with two examples each.**
> **Structured** — fixed schema, rows and columns, relational: SQL tables, CSV/Excel.
> **Semi-structured** — flexible, self-describing, hierarchical schema: JSON, XML (also YAML, HTML, logs).
> **Unstructured** — no predefined structure: free text, images/audio/video, PDFs.
> Extraction technique differs per class: pandas/SQL for structured; `json_normalize`/XPath for
> semi-structured; pdfplumber/OCR/regex/NER for unstructured.

**Q3. What is the curse of dimensionality? Give its effect on classification and clustering.**
> The phenomenon that **many types of data analysis become significantly harder as dimensionality
> increases**: **as dimensionality grows the data becomes increasingly sparse in the space it
> occupies**, so the observed objects are **quite possibly not a representative sample** of all
> possible objects.
> **Classification:** there are not enough data objects to build a model that reliably assigns a
> class to all possible objects.
> **Clustering:** the differences in **density** and in the **distances between points** — critical
> for clustering — **become less meaningful**.
> Remedy: dimensionality reduction (PCA, SVD) and feature subset selection.

**Q4. List the four data-quality problems and, for missing values, give the reasons and the handling strategies.**
> Problems: **noise, outliers, missing values, duplicate data**.
> Reasons for missing values: (a) **information is not collected** (people decline to give age/weight);
> (b) **attributes may not be applicable to all cases** (annual income for children).
> Handling: (1) **eliminate data objects**, (2) **estimate missing values**, (3) **ignore the missing
> value during analysis**, (4) **replace with all possible values weighted by their probabilities**.

**Q5. Your manager says "just delete all rows with missing values." Respond.**
> I would not act before checking the evidence. First, **verify how missingness is encoded** — in
> the Adult dataset it is the string **`?`**, so `isna()` reports zero until you convert it. Then:
> (1) compute **what percentage of rows** contain at least one missing value; (2) **compare the
> target distribution between rows with and without missing values**. If the incomplete rows differ
> systematically, deletion is **not MCAR** and would **change the population the dataset
> represents**, biasing every downstream conclusion. If the proportion is small and the missingness
> looks random, listwise deletion is defensible; otherwise **impute** — median for numerics (robust
> to skew/outliers), mode or an explicit `"Unknown"` category for categoricals, since "not reported"
> may itself be informative.

**Q6. Difference between an outlier and an error, with an example of each and how you treat them.**
> An **outlier** is a data object whose characteristics are **considerably different from most other
> objects** but which is **possible** — e.g. hours-per-week = 99 for an entrepreneur. An **error** is
> a value that is **impossible or contradicts the variable's definition** — e.g. age = −5 or 250,
> income = −20,000. **Treatment: correct or remove errors; for outliers, investigate first, then
> retain, cap (winsorize), or transform (log)** — deleting genuine extremes biases the model, and in
> fraud/intrusion detection the outliers *are* the signal.

**Q7. Explain the Data Science life cycle and why it is iterative.**
> [Ten stages — see 01-Lec1-2.] Iterative because: new data can reveal a **flawed assumption**; EDA
> may show the **target is poorly defined**; evaluation may expose **leakage or subgroup failures**;
> deployment can reveal **latency or usability constraints**; and feedback creates a new cycle of
> improvement. The pairs are bidirectional — problem definition ↔ data understanding,
> EDA ↔ feature engineering, modeling ↔ evaluation, deployment ↔ monitoring.

**Q8. Compare EDA and CDA.** → [see 5.5]. Add: EDA = Tukey, 1977; descriptive; hypothesis-generating.
CDA = inferential; hypothesis-testing. **EDA is not (or rarely) a definitive conclusion.**

**Q9. Compare a data warehouse and a data lake.** → [see 5.14 table]. Key line: **warehouse =
schema-on-write, processed/structured, BI and fast querying; lake = schema-on-read, raw/semi-/
unstructured, exploration and ML.**

**Q10. What is one-hot encoding, when do you use it, and when not?**
> It converts a categorical variable into **multiple binary columns, one per category**, with 1 =
> presence and 0 = absence — e.g. `Color` ∈ {Red, Blue, Green} becomes `Color_Red`, `Color_Blue`,
> `Color_Green`. Use it when the variable has **no intrinsic ordering** and the **number of categories
> is relatively small**. For **high-cardinality** variables the column explosion is impractical — use
> **target encoding or embeddings**. For **ordinal** variables use ordinal/label encoding so the order
> is preserved.

**Q11. Normalization vs Standardization — define both, give formulas, and say when each is used.**
> **Normalization** scales features to a **specific range** (e.g. [0,1]): x' = (x − min)/(max − min).
> **Standardization** rescales to **mean 0 and standard deviation 1**: z = (x − μ)/σ — "subtracting
> off the means and dividing by the standard deviation." Use normalization when you need a bounded
> range (neural nets, image pixels); standardization when the algorithm assumes roughly Gaussian,
> centred features (PCA, SVM, logistic regression, kNN). Tree-based models need neither.
> The terms are often used interchangeably in practice.

**Q12. Explain row-oriented vs columnar storage and why Parquet is preferred for analytics.**
> → [see 3.4 table]. Key terms to use: **projection pushdown**, **predicate pushdown**,
> **binary columnar compression (Snappy/ZSTD)**, vs CSV's **full-file scan** and **high parsing overhead**.

**Q13. What is data wrangling? List its steps.** → Definition + **Discovery, Cleaning, Structuring,
Enrichment, Validation, Publishing**. Structuring = reshaping wide ↔ long (pivot/melt).

**Q14. Differentiate supervised and unsupervised learning.** → [see 5.16 table]. Include the
formalism: supervised **f : ℝᵈ → {−1,+1}** (classifier) or **f : ℝᵈ → ℝ** (regressor);
unsupervised **f : ℝᵈ → {C₁,…,C_k}** (clusters).

**Q15. Given these scenarios, name the data science task and one algorithm.**
| Scenario | Task | Algorithm |
|---|---|---|
| Detect fraudulent credit-card transactions | **Anomaly detection** | Distance-based / density-based |
| Predict next year's unemployment rate | **Regression** | Linear regression |
| Forecast monthly sales for the next quarter | **Time series** | **ARIMA** / exponential smoothing |
| Group customers into segments from transaction data | **Clustering** | **k-Means / DBSCAN** |
| Find cross-selling opportunities from purchase baskets | **Association analysis** | **Apriori / FP-Growth** |
| Bucket a new customer into a known customer group | **Classification** | Decision tree / kNN / naive Bayes |

---

## SECTION C — Applied / long answers

**Q16. You are given this table. Identify every data-quality issue and state your action with justification.**

| ID | Name | Age | Gender | Income (PKR) |
|---|---|---|---|---|
| 1 | Ali | 25 | M | 50,000 |
| 2 | Sara | | Female | 60000 |
| 3 | Ahmed | 250 | Male | -20000 |
| 8 | Kamran | 40 | M | ? |
| 9 | Ali | 25 | Male | 50,000 |
| 10 | Bilal | -5 | M | 55,000 |
| 15 | Zainab | 27 | female | null |

> **Answer:** use the **Problem → Evidence → Action → Justification** format from Activity-1 —
> the full seven-issue table is in `04-Lec5-6-Scrubbing-Wrangling.md §4.2`. Cover: missing values,
> **disguised missing values (`?`, `null` as strings)**, impossible values (age 250/−5, income
> −20,000) which are **errors not outliers**, inconsistent gender categories (M/Male/MALE/F/Female/
> female), inconsistent numeric formatting (comma thousands separators forcing an object dtype),
> **duplicates that `drop_duplicates()` misses because the surrogate ID differs**, and wrong dtypes.

**Q17. Write a reproducible scikit-learn preprocessing pipeline for a dataset with numeric columns
[age, bmi, glucose] and categorical columns [gender, region], and justify every choice.**
> → Code in §4.8, justification table immediately after it. Must mention **`handle_unknown="ignore"`**
> and **leakage prevention** (fit on train only, transform on test).

**Q18. Given the Lec-7 activity table of 12 records, answer the six analysis questions.**
> → Full worked answers with the arithmetic in §5.9. The two most predictive features are
> **hours-per-week (55 vs 29 hours on average)** and **marital status**.

**Q19. Which EDA technique would you use for each objective?**
| Objective | Technique |
|---|---|
| Idea of the distribution of a variable | **Histogram** |
| Finding outliers | **Histogram, scatterplots, box-and-whisker plots** |
| Quantify the relationship between two variables | **2D scatter plot + curve fitting; covariance and correlation** |
| Two exposure variables and one outcome | **Heatmap** |
| High-dimensional data | **t-SNE or PCA + 2D/3D scatterplot** |
| Comparing multiple groups | **Side-by-side boxplot** |
| Categorical data | **Descriptive statistics / frequency tabulation** |

**Q20. Explain PCA. What is it optimizing, and how does it differ from LDA?**
> PCA finds **new axes (principal components) that are linear combinations of the original features,
> ordered by the amount of variance each captures**; PC1 captures the most variance and each
> subsequent component is **orthogonal** to the previous ones. Its **goal is to find a projection
> that captures the largest amount of variation in the data**, transforming a large set of variables
> into a smaller one that still contains most of the information.
> It is **linear, unsupervised, fast, deterministic and interpretable**, preserves **global
> structure**, and provides a `transform()` so it generalizes to unseen data — but it **cannot
> capture nonlinear structure**.
> **LDA** is also linear but **supervised**: it uses the class labels to find axes that **maximize
> between-class separation while minimizing within-class variance**. Its number of components is
> **capped at (number of classes − 1)** and it assumes classes are roughly **Gaussian with equal
> covariance**. In short: **PCA optimizes variance; LDA optimizes class separability.**

**Q21. What is data leakage and how do you prevent it?**
> Leakage is when information that would **not be available at prediction time** is used during
> training, inflating validation scores and collapsing in production. Sources: (a) using a feature
> computed from the future or from the target; (b) **fitting imputers/scalers/encoders on the full
> dataset before the train/test split**, so test statistics leak into training; (c) duplicated rows
> spanning both splits; (d) identifiers that proxy the target (the Adult dataset's **`fnlwgt`** census
> weight is a related trap — it is a sampling weight, not a legitimate predictor).
> Prevention: **split first, then fit all transformers inside a `Pipeline`/`ColumnTransformer` on the
> training fold only**; deduplicate before splitting; use time-based splits for temporal data; and
> check that every feature is **available at prediction time**.

**Q22. Why sample at all, and what makes a sample good?**
> Statisticians sample because **obtaining** the entire data set of interest is too expensive or time
> consuming; data mining samples because **processing** it is. The key principle: **using a sample
> will work almost as well as using the entire data set if the sample is representative** — i.e. if
> it has approximately the same property of interest as the original data. Sample size matters: the
> lecture's 8000 → 2000 → 500 point illustration shows structure degrading as the sample shrinks.
> For an **imbalanced target**, use **stratified sampling** so the minority class is preserved.

**Q23. Explain the three regression-fit curves in Lecture 8.**
> The **green straight line** is a simple linear fit that may **underfit** (high bias — too simple to
> capture the trend). The **blue smooth curve** follows the underlying trend — a **good fit**. The
> **red wiggly curve** passes through every training point: it has **overfit**, modelling the noise
> rather than the signal, so it will generalize poorly to new data. This is the **bias–variance
> trade-off**, and it motivates validation on held-out data (life-cycle stage 8).

**Q24. A colleague says the Adult data shows that being male *causes* higher income, since 30% of
men vs 10% of women earn >50K. Respond.**
> Those figures are correct (overall 24%, men 30%, women 10%) but they establish **association, not
> causation**. This is an **EDA** finding, and **EDA is not a definitive conclusion — it generates
> hypotheses**. The gap is likely **confounded** by hours-per-week, occupation, education and
> marital status, all of which differ by sex in this census data; it also reflects **when and how
> the data were collected** (1994 US census extract), so it may not generalize. To move from
> exploratory to confirmatory you would need to control for the confounders (e.g. stratify or fit a
> model with those covariates) and state the assumptions — that is **CDA / inferential statistics**,
> not EDA. Also note the **target is imbalanced (76/24)**, so any classifier trained on it must be
> evaluated with precision/recall rather than accuracy.
