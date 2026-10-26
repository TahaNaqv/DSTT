# ⚡ CRAM SHEET — read this in the last hour

## The lists you must be able to write instantly

**Big Data 3 V's:** Volume · Velocity · Variety & Complexity  (+Veracity, +Value = 5 V's)

**DS life cycle (10):** Problem Understanding → Data Acquisition → Data Understanding → Data Cleaning
→ EDA → Feature Engineering → Modeling → Evaluation → Deployment → Monitoring & Maintenance

**DS challenges (7):** Scalability · High Dimensionality · Complex & Heterogeneous Data ·
Data Ownership & Distribution · Privacy Preservation · Streaming Data · Data Quality

**DS tasks (6):** Classification · Regression · Anomaly detection · Time series · Clustering · Association analysis

**Predictive:** Classification, Regression, Anomaly Detection.  **Descriptive:** Clustering,
Association Rule Discovery, Sequential Pattern Discovery. *(per Dr. Mateen's slide)*

**Levels of measurement (NOIR):** Nominal · Ordinal · Interval · Ratio

**Attribute types:** Nominal · Binary (symmetric / asymmetric) · Ordinal · Numeric (interval-scaled / ratio-scaled)

**Characteristics of structured data (4):** Dimensionality · Sparsity · Resolution · Distribution

**Data quality problems (4):** Noise · Outliers · Missing values · Duplicate data

**Missing-value handling (4):** Eliminate data objects · Estimate (impute) · Ignore during analysis ·
Replace with all possible values weighted by probabilities

**Preprocessing techniques (7):** Aggregation · Sampling · Dimensionality Reduction ·
Feature subset selection · Feature creation · Discretization & Binarization · Attribute Transformation

**Preprocessing STEPS (4):** Cleaning → Transformation → Reduction → Integration

**Sampling types:** Simple random (with / without replacement) · Stratified

**Feature subset selection (4):** Brute-force · Embedded · Filter · Wrapper

**Feature engineering branches (4):** Selection · Extraction · Construction · Encoding

**Data wrangling steps (6):** Discovery → Cleaning → Structuring → Enrichment → Validation → Publishing

**Data filtering types (5):** Basic · Advanced · Date/Time · Text-based · Statistical

**Dim. reduction (5):** PCA · t-SNE · UMAP · LDA · Autoencoders

**Database families (4):** relational · document-oriented · key-value · graph-oriented

**ML flow (5):** Problem Nature Identification → Training Data Collection → Model Training →
Model Validation → Model Deployment

---

## One-line discriminators (the questions that separate an A from a B)

| Pair | The distinction |
|---|---|
| **Interval vs Ratio** | Ratio has a **true/inherent zero**, so ratios are meaningful. °C/°F = interval; Kelvin = ratio. 20 °C is not twice 10 °C; 20 K is twice 10 K. |
| **Nominal vs Ordinal** | Ordinal has a **meaningful order** but the **magnitude between successive values is unknown**. |
| **Discrete vs Continuous** | Discrete = **finite** set of values (int); continuous = **real numbers** (float). Binary is a special case of discrete. |
| **Symmetric vs Asymmetric binary** | Symmetric = both outcomes equally important (gender). Asymmetric = not (medical test); **assign 1 to the most important outcome**. |
| **Noise vs Outlier** | Noise = **modification of original values** (random error). Outlier = a **legitimate object with very different characteristics** — may be the most valuable point. |
| **Outlier vs Error** | Outlier = unusual but **possible** (99 hrs/week). Error = **impossible/contradicts the domain** (age = −5). Fix errors; investigate outliers. |
| **Normalization vs Standardization** | Normalization → a **specific range** [0,1]. Standardization → **mean 0, SD 1** (subtract mean, divide by SD). *Terms are often used interchangeably.* |
| **One-hot vs Label encoding** | One-hot for **nominal** (no order); label encoding invents a false ordering, so use it only for **ordinal** data or the target. |
| **Feature selection vs extraction** | Selection **keeps a subset of original** features (interpretable). Extraction **creates new** combined features (PCA, autoencoders). |
| **PCA vs LDA** | Both linear. PCA = **unsupervised**, maximizes **variance**. LDA = **supervised**, maximizes **class separability**; components capped at (classes − 1). |
| **t-SNE vs UMAP** | Both nonlinear for visualization. UMAP is **faster**, preserves **more global structure**, and **can transform new data**; t-SNE cannot and its **inter-cluster distances are meaningless**. |
| **EDA vs CDA** | EDA **finds a good description** and **raises** questions → **descriptive** statistics. CDA **tests a hypothesis** and **settles** questions → **inferential** statistics. |
| **Covariance vs Correlation** | Correlation = **standardized covariance**, bounded [−1,+1] and unitless. Covariance is unbounded and unit-dependent. Both capture only **linear** association. |
| **Warehouse vs Lake** | Warehouse = **processed + structured, schema-on-WRITE**, BI/fast queries. Lake = **raw + any format, schema-on-READ**, exploration & ML. |
| **Row vs Columnar format** | CSV row-oriented: **full scan to read one column**. Parquet columnar: **projection + predicate pushdown**, binary compression, built for OLAP. |
| **Supervised vs Unsupervised** | Labels present vs absent. Classification/regression vs clustering. f: ℝᵈ→{−1,+1} or ℝ, vs f: ℝᵈ→{C₁…C_k}. |
| **Classification vs Regression** | y **discrete** (class) vs y **real-valued**. Classifier vs **regressor**. |
| **Traditional programming vs ML** | Programming: Data + **Program** → Output. ML: Data + **Output** → **Program**. |
| **Underfit / good fit / overfit** | Green straight line / blue smooth curve / red wiggly curve through every point. |
| **Why statisticians vs data miners sample** | Statisticians: **obtaining** all the data is too expensive. Data mining: **processing** all the data is too expensive. |

---

## Numbers to remember

| Fact | Value |
|---|---|
| Google processing | **20 PB/day** |
| Facebook daily logs | **60 TB** |
| eBay | **6.5 PB** + 50 TB/day |
| 1000 Genomes | **200 TB** |
| Cost of 1 TB disk / time to read it | **$35 / 3 hours at 100 MB/s** |
| Adult dataset shape | **(32561, 15)** — 32,561 records, 15 features |
| Adult `adult.test` | **~16,281 records** |
| Adult: unknown country count | **? = 583** |
| Adult high-income rates | **people 24% · men 30% · women 10%** |
| McKinsey 2011 forecast | **140,000–190,000 predictive analysts; 1.5M managers/analysts by 2018** |
| Sparse data | often **< 1% non-zero** |
| Tukey's *Exploratory Data Analysis* | **1977** |
| IQR outlier fence | **Q1 − 1.5·IQR, Q3 + 1.5·IQR** |
| LDA max components | **number of classes − 1** |

---

## Code snippets you may be asked to write

```python
# Load and inspect
df = pd.read_csv('data.csv'); df.shape; df.info(); df.describe(); df.head()

# Missing values — including DISGUISED ones
df = df.replace('?', np.nan)                 # Adult dataset uses '?' not NaN
df.isna().sum()                              # count per column
df.isna().any(axis=1).sum()                  # rows with >= 1 missing
df['Age'].fillna(df['Age'].median(), inplace=True)

# Categorical cleanup
df['sex'] = df['sex'].str.strip().str.lower()
df['sex'].value_counts()

# Duplicates
df.duplicated().sum(); df.drop_duplicates(inplace=True)

# Filtering  (use & | ~, NOT and/or/not, and parenthesise each condition)
df[(df['column1'] > 50) & (df['column2'] == 'Yes')]
df.query('age > 25 and salary > 70000')

# Group-by / aggregation
df.groupby('department')['salary'].mean()
df.groupby('country').size()

# Encoding & mapping
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
pd.get_dummies(df, columns=['region'])

# Outliers via IQR
Q1, Q3 = df['x'].quantile([.25,.75]); IQR = Q3 - Q1
mask = (df['x'] < Q1 - 1.5*IQR) | (df['x'] > Q3 + 1.5*IQR)

# The reproducible pipeline
num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale",  StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                     ("encode", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", num_pipe, num_cols),
                                  ("cat", cat_pipe, cat_cols)])
X_clean = preprocessor.fit_transform(df)
```

---

## Quotable lines (drop one into a long answer and it reads like you were in class)

- *"Big Data is any data that is **expensive to manage and hard to extract value from**."*
- *"More data does not automatically mean better decisions."*
- *"**Garbage In, Garbage Out** — model quality is bounded above by input data quality."*
- *"**Cleaning decisions can change the meaning of the dataset — document them.**"*
- *"**EDA should generate hypotheses, not merely produce attractive charts.**"*
- *"Using a sample will work almost as well as using the entire data set, **if the sample is representative**."*
- *"Sometimes **less is more** — this is the case with aggregation."*
- *"**The ability to iterate is a core professional skill.**"*
- *"EDA is **NOT (or rarely) a definitive conclusion**."*
- *"Avoid leakage: **features must be available at prediction time**."*
