# Part 4 — Lectures 5 & 6: Data Preparation, Scrubbing, Wrangling, Preprocessing, Feature Engineering

---

## 4.1 Data Preparation — the 4 tasks (instructor's speaker notes, learn verbatim-ish)

> *"One of the first tasks when analyzing data is to collect and prepare the data in a format
> appropriate for analysis of the samples."*

| # | Task | Detail from the notes |
|---|---|---|
| 1 | **Obtaining the data** | Data can be **read directly from a file** or **obtained by scraping the web**. |
| 2 | **Parsing the data** | The right parsing procedure **depends on what format the data are in**: plain text, fixed columns, CSV, XML, HTML, etc. |
| 3 | **Cleaning the data** | Survey responses and other data files are **almost always incomplete**. Sometimes there are **multiple codes** for things such as *not asked*, *did not know*, and *declined to answer*. And there are almost always errors. **A simple strategy is to remove or ignore incomplete records.** |
| 4 | **Building data structures** | Store the data in a structure that lends itself to the analysis. **If the data fit into memory, building a data structure is usually the way to go. If not, a database is built — an out-of-memory data structure.** Most databases provide a **mapping from keys to values, so they serve as dictionaries.** |

**The pandas DataFrame** — *"a two-dimensional, size-mutable, potentially heterogeneous tabular
data structure with labeled axes."* Learn that definition word for word; it's a favourite.

---

## 4.2 ★ The dirty-table example (Lecture 5, slide 4) — MEMORISE THIS ANALYSIS

| ID | Name | Age | Gender | Income (PKR) |
|---|---|---|---|---|
| 1 | Ali | 25 | M | 50,000 |
| 2 | Sara | *(blank)* | Female | 60000 |
| 3 | Ahmed | **250** | Male | **-20000** |
| 4 | Sara | 30 | F | 60000 |
| 5 | John | 28 | **MALE** | 45,000 |
| 6 | Maryam | 35 | **female** | 70,000 |
| 7 | Ayesha | 29 | F | 75,000 |
| 8 | Kamran | 40 | M | **?** |
| 9 | Ali | 25 | Male | 50,000 |
| 10 | Bilal | **-5** | M | 55,000 |
| ... | ... | ... | ... | ... |
| 14 | Fatima | **200** | F | 72,000 |
| 15 | Zainab | 27 | female | **null** |
| 20 | Maria | *(blank)* | F | 80,000 |

### Every issue, and the fix (this is the exam answer)

| # | Issue | Evidence | Action | Justification |
|---|---|---|---|---|
| 1 | **Missing values** | Age blank for IDs 2, 20 | Impute with **median** age | Median is robust to the outliers already present; deleting 2/20 rows loses 10% of the data |
| 2 | **Disguised missing values** | Income = `?` (ID 8) and `null` (ID 15) — strings, not NaN | Convert `?`, `null`, `""` → `np.nan` **before** counting missingness | `df.isna()` reports **0 missing** while the placeholder is a string — a false clean bill of health |
| 3 | **Impossible / erroneous values** | Age = **250, 200, −5**; Income = **−20000** | These are **errors, not outliers** → set to NaN and impute, or drop | Age can't be negative or 250; income can't be −20,000 PKR. Violates the domain definition |
| 4 | **Inconsistent categories** | Gender = M, Male, MALE, F, Female, female | Standardize: `.str.strip().str.lower()` then map → {M, F} or {0, 1} | Six labels for two real categories inflates cardinality and breaks group-by/encoding |
| 5 | **Inconsistent numeric formatting** | "50,000" (with comma) vs 60000 (no comma) | Strip commas, cast to numeric: `.str.replace(',','').astype(float)` | Commas force the whole column to load as **object/text**, so no arithmetic works |
| 6 | **Duplicate records** | ID 1 vs ID 9 (Ali, 25, M, 50,000); ID 4 vs 18 (Sara); ID 5 vs 12 (John) | `df.drop_duplicates(subset=[...])` on the business key, not the surrogate ID | The ID differs, so `drop_duplicates()` on the whole row misses them. Duplicates inflate patterns |
| 7 | **Wrong data types** | Age and Income read as `object` because of blanks/`?`/commas | Cast after cleaning | Numeric ops, `describe()` and scalers all fail on object columns |

> **Say this in the exam:** the ID column is a **surrogate key**, so it differs even for genuine
> duplicates. Always deduplicate on the **business key** (Name + Age + Gender + Income), never on
> the row including the ID.

---

## 4.3 The Adult dataset (UCI) — the running example

- Hosted at UCI ML Repository: `https://archive.ics.uci.edu/ml/datasets/Adult`
- **~32,000 observations** (exactly **32,561 rows × 15 columns**, `df.shape` → `(32561, 15)`)
- Financial/demographic parameters of the US population: age, sex, marital, country, **income**
  (Boolean: whether the person makes **more than $50,000 per annum**), education, occupation,
  capital gain, etc.
- Driving question: ***"Are men more likely to become high-income professionals than women, i.e.
  to receive an income of over $50,000 per annum?"***

### Files in the download

| File | Size | Purpose | Contains | Used for ML? |
|---|---|---|---|---|
| `adult.data` | 3.8 MB | **Training data** | ~32,561 records, 14 features + income target | **Yes** |
| `adult.test` | 1.9 MB | **Test data** | ~16,281 records, 14 features + income target | **Yes** |
| `adult.names` | 5.1 KB | Dataset documentation | Attribute descriptions, possible values, missing-value info, background | Reference |
| `old.adult.names` | 4.2 KB | Older documentation | Earlier version of the description/metadata | Usually no |
| `Index` | 140 B | Repository/index information | File/index metadata | No |

### The 15 columns
`age, type_employer (workclass), fnlwgt, education, education_num, marital, occupation,
relationship, race, sex, capital_gain, capital_loss, hr_per_week, country, income`

> **`fnlwgt` is a census sampling weight — it is NOT a predictive feature. Dropping it is a
> standard, defensible answer.** (This is exactly the leakage-adjacent trap in your Assignment-1.)

### The code walked through in class

```python
file = open('files/ch03/adult.data', 'r')
def chr_int(a):
    if a.isdigit(): return int(a)
    else: return 0

data = []
for line in file:
    data1 = line.split(', ')
    if len(data1) == 15:
        data.append([chr_int(data1[0]), data1[1], chr_int(data1[2]), data1[3],
                     chr_int(data1[4]), data1[5], data1[6], data1[7], data1[8],
                     data1[9], chr_int(data1[10]), chr_int(data1[11]),
                     chr_int(data1[12]), data1[13], data1[14]])

import pandas as pd
df = pd.DataFrame(data)
df.columns = ['age','type_employer','fnlwgt','education','education_num','marital',
              'occupation','relationship','race','sex','capital_gain','capital_loss',
              'hr_per_week','country','income']

df.shape                      # (32561, 15)  -> 32,561 records, 15 features

counts = df.groupby('country').size()
print(counts.head())
# country
# ?          583      <-- the FIRST row is the number of samples with UNKNOWN country
# Cambodia    19
# Canada     121
# China       75
# Columbia    59
```

> **`?` = 583** is the point of that slide: in the Adult dataset **missing values are encoded as
> the string `?`, not as NaN**. That is why `isna()` initially reports zero.

### Splitting by group and computing rates
```python
ml  = df[(df.sex == 'Male')]
ml1 = df[(df.sex == 'Male')   & (df.income == '>50K\n')]
fm  = df[(df.sex == 'Female')]
fm1 = df[(df.sex == 'Female') & (df.income == '>50K\n')]

df1 = df[(df.income == '>50K\n')]
print('The rate of people with high income is: ', int(len(df1)/float(len(df))*100), '%.')
print('The rate of men   with high income is: ', int(len(ml1)/float(len(ml))*100), '%.')
print('The rate of women with high income is: ', int(len(fm1)/float(len(fm))*100), '%.')
```
**Output — learn these three numbers:**
```
The rate of people with high income is: 24 %.
The rate of men   with high income is: 30 %.
The rate of women with high income is: 10 %.
```
> **Interpretation (the answer to the driving question): yes — 30% of men vs 10% of women earn
> >$50K, a 3× gap. Also note the overall 24/76 split = the target is IMBALANCED.**
>
> ⚠️ Caution for a discussion question: this is **association, not causation**. The gap may be
> confounded by hours-per-week, occupation and education.

> **For categorical data, a simple tabulation of the frequency of each category is the best
> non-graphical exploration.** (Instructor's note — good one-liner to quote.)

---

## 4.4 Data Scrubbing (= Data Cleaning)

> **Definition: the process of detecting, correcting, or removing corrupt or inaccurate records from a dataset.**

**Goals (3):**
1. Improve **data quality and reliability**
2. Ensure **consistency and accuracy**
3. **Prepare data for analysis**

**Importance:**
- **Data Quality** — clean data ensures more accurate and reliable results
- **Efficiency** — reduces processing time and storage needs

**Common issues (5):**
| Issue | Example |
|---|---|
| **Missing values** | Data points not recorded |
| **Inconsistent data** | Different formats or units — **"US" vs "USA"** |
| **Outliers** | Values significantly different from others |
| **Duplicates** | Multiple entries of the same data point |
| **Incorrect data types** | Numeric data stored as text |

---

## 4.5 DATA WRANGLING

> **Definition: the process of cleaning, restructuring, and enriching raw data into a desired format
> for better decision-making and analysis.**
> **Why?** Raw data is **messy, inconsistent, and incomplete**. Wrangling ensures **data quality,
> consistency, and usability.**

### ★ The 6 steps in data wrangling (memorise the order — very likely a direct question)

| # | Step | What it means |
|---|---|---|
| 1 | **Data Discovery** | Understanding **sources and structure** |
| 2 | **Data Cleaning** | Handling missing values, duplicates, errors |
| 3 | **Data Structuring** | **Reshaping datasets (wide ↔ long format)** — pivot / melt |
| 4 | **Data Enrichment** | Adding **external or derived features** |
| 5 | **Validation** | Ensuring **correctness and consistency** |
| 6 | **Publishing** | Making it **ready for analysis / ML models** |

Mnemonic: **D-C-S-E-V-P** → "**D**iscover, **C**lean, **S**tructure, **E**nrich, **V**alidate, **P**ublish."

Slide diagram **Raw → Wrangled → Ready**:
- *Raw inputs*: messy table (alice/BOB, `NaN` age, `05/12/25` vs `12 May 2025`), text/log snippet, un-normalized API JSON
- *Wrangling steps*: **Clean** (trim case, parse dates, impute missing Age→median) → **Transform**
  (encode categories, scale numbers, log-transform skew) → **Join** (merge customers ↔ transactions
  on `user_id`) → **Reshape** (wide ↔ long, pivot/melt) → **Validate** (ranges, duplicates,
  referential integrity)
- *Clean outputs*: normalized table; export to **CSV/Parquet, JSON/NDJSON, XML, DB tables**;
  downstream to **EDA & dashboards, ML training/serving, APIs & apps**
- Tools shown: **Python (pandas), R (dplyr/tidyr), PySpark**

### Tools for data wrangling (know which is for what scale)

| Tool | Role |
|---|---|
| **Pandas** | Handling missing values, reshaping, merging, groupby |
| **NumPy** | Efficient numerical operations |
| **Dask** | **Parallel/distributed** wrangling for large datasets |
| **PySpark** | **Scalable** wrangling on big data |
| **Apache Spark** (PySpark, SparkR) | Distributed large-scale wrangling |
| **Hadoop** | Preprocessing **raw unstructured** data |
| **Databricks** | Unified environment for **wrangling + ML** |

### Wrangling examples (6, each with its one-line example)
| Operation | Example |
|---|---|
| Handling missing data | Replace NaN in `Age` with mean/median |
| Removing duplicates | Duplicate customer IDs removed from database |
| Standardizing formats | Convert `"12-05-25"` → `"2025-05-12"` |
| Encoding categorical variables | Male/Female → 0/1 |
| Merging datasets | Joining `Sales.csv` with `Customer.csv` |
| Outlier treatment | Replace extreme income values with **capped** values (winsorizing) |

---

## 4.6 Data Filtering

> **Definition: the process of selecting a subset of data based on specific conditions or criteria.**

**Purpose (3):** remove irrelevant/unnecessary data · focus on specific data points for analysis ·
improve data quality and relevance.

### Types (5)
| Type | Description |
|---|---|
| **Basic Filtering** | Simple conditions — by value, range, or category |
| **Advanced Filtering** | Multiple conditions & complex queries — multiple columns, logical operators |
| **Date and Time Filtering** | Selecting data within specific time frames or periods |
| **Text-Based Filtering** | Filtering on keywords, patterns, or text length |
| **Statistical Filtering** | Filtering on statistical properties — mean, standard deviation, **quantiles** |

```python
# Basic: condition-based
filtered_df = df[df['column_name'] > 50]

# Advanced: multiple conditions with & and |
filtered_df = df[(df['column1'] > 50) & (df['column2'] == 'Yes')]
```
> ⚠️ **In pandas use `&`, `|`, `~` — NOT `and`, `or`, `not` — and each condition MUST be wrapped in
> parentheses** because `&` binds tighter than `>`. A near-certain MCQ/code-error question.

---

## 4.7 DATA PREPROCESSING — *"Garbage In, Garbage Out"*

**Why preprocess (5 reasons):**
1. Real-world data is **incomplete, noisy, and inconsistent**
2. **Model quality is bounded above by input data quality**
3. Preprocessing **improves accuracy, reduces training time**
4. Ensures **compatibility with algorithmic assumptions** (e.g. scale, distribution)
5. **Reduces the risk of biased or misleading conclusions**

**Common data quality issues (5):**
| Issue | Manifestation |
|---|---|
| Missing values | Null / NaN entries, blank fields |
| Noise & outliers | Measurement errors, extreme values |
| Inconsistency | Mismatched units, formats, categories |
| Duplicates | Repeated records **inflating patterns** |
| **Imbalanced classes** | **Skewed target distributions** |

### ★ The 4 STEPS in data preprocessing (memorise the order)

| # | Step | Contents |
|---|---|---|
| 1 | **Data Cleaning** | Handle missing values, remove duplicates, fix noisy/erroneous entries |
| 2 | **Data Transformation** | **Normalize / standardize, encode categoricals, discretize continuous variables** |
| 3 | **Data Reduction** | **Reduce volume while preserving structure** — sampling, aggregation, **PCA** |
| 4 | **Data Integration** | **Combine multiple sources into a coherent, conflict-free dataset** |

### Common tools
| Tool | Used for |
|---|---|
| **pandas** | `isna()`, `fillna()`, `drop_duplicates()` |
| **scikit-learn** | `SimpleImputer`, `StandardScaler`, `OneHotEncoder` |
| **OpenRefine** | **GUI-based** cleaning of messy data |
| **PyOD / IsolationForest** | **Outlier & anomaly detection** |

---

## 4.8 ★ THE PIPELINE CODE — be able to write this from memory

```python
import pandas as pd
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose       import ColumnTransformer
from sklearn.pipeline      import Pipeline

df = pd.read_csv("patients.csv")
df.drop_duplicates(inplace=True)

num_cols = ["age", "bmi", "glucose"]
cat_cols = ["gender", "region"]

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

X_clean = preprocessor.fit_transform(df)
```

**Be ready to justify every line:**
| Choice | Justification |
|---|---|
| `strategy="median"` for numerics | **Robust to outliers and skew** — the mean gets dragged by extremes |
| `strategy="most_frequent"` for categoricals | The only sensible central tendency for nominal data is the **mode** |
| `StandardScaler` | Puts features on a comparable scale → required by **distance-based (kNN, k-means, SVM) and gradient-based** algorithms; trees don't need it |
| `OneHotEncoder` | Nominal categories have **no intrinsic order**; label-encoding them would invent a false ordering |
| `handle_unknown="ignore"` | **Previously unseen categories at prediction time do not crash the pipeline** — they encode as all-zeros |
| `ColumnTransformer` + `Pipeline` | **Reproducibility**, and it **prevents data leakage**: statistics (median, mean, σ) are learned on `fit` (train) only and merely `transform`ed onto test data |

> **Leakage soundbite:** *"If you impute or scale before the train/test split, information from the
> test set leaks into training via the imputation statistics and scaling parameters. Fitting inside
> a Pipeline after the split prevents this."*

---

## 4.9 Data Transformation

| Technique | Definition |
|---|---|
| **Normalization** | Scaling features to a **specific range** (e.g. [0, 1] or [−1, 1]) |
| **Standardization** | Scaling features to have **mean 0 and standard deviation 1** |
| **Encoding categorical variables** | **One-hot encoding, Label encoding** |

### ★ One-Hot Encoding (instructor's own notes — learn the worked example)

> *One-hot encoding converts categorical data into a numerical format that ML algorithms can use.
> Particularly useful for categorical variables that **do not have an inherent order**.*

Original `Color` column: Red, Blue, Green, Blue, Red →

| Color_Red | Color_Blue | Color_Green |
|---|---|---|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |

*Each original category is represented as a binary vector, with 1 = presence and 0 = absence.*

**When to use one-hot encoding:**
- When the categorical variable has **no intrinsic ordering** (colour, city names, etc.)
- When **the number of categories is relatively small.** For very large numbers of categories, use
  **target encoding or embeddings** instead.

**Label encoding** maps categories to integers (Red→0, Blue→1, Green→2). **Use it only for ordinal
data** (small<medium<large) or for the **target** — using it on nominal features invents a false
ordering and a false notion of distance.

---

## 4.10 FEATURE ENGINEERING

> **Definition: using domain knowledge to construct new variables that make patterns easier for a
> model to learn.**

### The 4 branches (learn which technique sits under which branch)

| Branch | Techniques |
|---|---|
| **Feature Selection** | **Filter** (correlation, chi-square), **Wrapper** (RFE), **Embedded** (Lasso, tree importances) |
| **Feature Extraction** | **PCA, autoencoders, text embeddings** — compress raw inputs into compact representations |
| **Feature Construction** | **Ratios, interaction terms, date-part extraction, aggregations, domain formulas** |
| **Feature Encoding** | **One-hot, label, target/mean encoding, binning, log/Box-Cox transforms** |

> ⚠️ **Selection vs Extraction** — selection *keeps a subset of the original* features (interpretable);
> extraction *creates new* features as combinations of the originals (compact, less interpretable).

**Tools:** **Featuretools** (automated feature synthesis / **Deep Feature Synthesis, DFS**) ·
**tsfresh** (automatic feature extraction from time series) · **category_encoders** (target, WOE,
hashing, ordinal encoders).

**Key reading cited:** Zheng & Casari, *Feature Engineering for Machine Learning*, O'Reilly 2018 ·
Kanter & Veeramachaneni, *Deep Feature Synthesis*, IEEE DSAA 2015 (basis for Featuretools).

### Constructing new features — code
```python
import pandas as pd, numpy as np

df["bmi"]        = df["weight_kg"] / (df["height_m"] ** 2)          # domain formula
df["age_bucket"] = pd.cut(df["age"], bins=[0,18,35,60,100],
                          labels=["teen","adult","mid","senior"])   # binning/discretization

df["order_date"]  = pd.to_datetime(df["order_date"])
df["order_month"] = df["order_date"].dt.month                        # date-part extraction
df["is_weekend"]  = df["order_date"].dt.dayofweek >= 5

df["log_income"]  = np.log1p(df["income"])                           # log transform for skew

# Automated feature synthesis
import featuretools as ft
es = ft.EntitySet(id="orders")
es = es.add_dataframe(dataframe_name="orders", dataframe=df,
                      index="order_id", time_index="order_date")
features, feature_defs = ft.dfs(entityset=es, target_dataframe_name="orders", max_depth=2)
```
> `np.log1p(x) = log(1+x)` — used instead of `log(x)` because it **handles zeros safely** and tames
> right-skewed variables like income. `dayofweek >= 5` → Saturday(5)/Sunday(6).

---

## 4.11 DATA INTEGRATION — merging & resolving conflicts

### Merge strategies (4)
| Strategy | Meaning |
|---|---|
| **Inner / Outer / Left / Right join** | Combine tables on **shared keys** (SQL JOIN, `pandas.merge`) |
| **Concatenation** | **Stack datasets with the same schema** (`pandas.concat`) |
| **Entity resolution / record linkage** | Match records referring to **the same real-world entity** across sources |
| **Schema matching** | **Align differing column names, types, and units** before merging |

Join cheat-sheet: **inner** = only matching keys · **left** = all left rows + matches ·
**right** = all right rows + matches · **outer** = all rows from both, NaN where unmatched.

### Resolving conflicts (4 conflict types)
| Conflict | Resolution |
|---|---|
| **Value conflicts** | Same entity, different values (e.g. two ages for one patient) → **rule-based or trust-score resolution** |
| **Unit / format mismatch** | Convert to a **common unit & format** (kg vs lb, date formats) **before** merging |
| **Duplicate entities** | **Fuzzy matching (Levenshtein, Jaro-Winkler)** to detect near-duplicate records |
| **Structural conflicts** | Different schemas for the same concept → **harmonize via a canonical data model** |

### Code
```python
import pandas as pd
patients = pd.read_csv("patients.csv")      # source A
labs     = pd.read_csv("lab_results.csv")   # source B

# 1. Harmonize keys/units BEFORE the merge
labs["patient_id"]   = labs["pid"].astype(str)
labs["glucose_mgdl"] = labs["glucose_mmol"] * 18.0

# 2. Merge on the common key
merged = patients.merge(labs, on="patient_id", how="left")

# 3. Resolve value conflicts: prefer the most recent record
merged = (merged.sort_values("test_date")
                .drop_duplicates("patient_id", keep="last"))

# 4. Fuzzy match near-duplicate names across sources
from rapidfuzz import fuzz
score = fuzz.token_sort_ratio("Jon Smith", "John Smith")   # 90+
```

**Key reading cited:** Doan, Halevy & Ives, *Principles of Data Integration* (Morgan Kaufmann 2012) ·
Elmagarmid, Ipeirotis & Verykios, *Duplicate Record Detection: A Survey* (IEEE TKDE 2007) ·
Dong & Srivastava, *Big Data Integration* (2015).

---

## 4.12 ★ DIMENSIONALITY REDUCTION — the 5-method comparison (instructor wrote detailed notes = likely exam question)

> Purpose: **reduce the number of features while preserving structure — fights the curse of
> dimensionality, speeds up training, and enables visualization.**

| Method | Linear? | Supervised? | What it does | Strengths | Limitations / When to use |
|---|---|---|---|---|---|
| **PCA** | **Linear** | Unsupervised | Finds new axes (**principal components**) that are **linear combinations of the original features, ordered by variance captured**. PC1 captures the most variance; PC2 the next most, **orthogonal** to PC1 | **Fast, deterministic, interpretable** (components are weighted sums). **Preserves global structure** well. Has `transform()` so it generalizes to new data | **Only captures straight-line relationships**; misses nonlinear patterns (spiral/manifold). Use for quick reduction, removing **multicollinearity**, speeding up downstream models |
| **t-SNE** | Nonlinear | Unsupervised | Converts **similarities between points into probabilities**, then arranges points in 2D/3D so similar points stay close | **Excellent at revealing local clusters** | **Distances between clusters and cluster sizes are NOT meaningful** — only relative neighbourhoods. Computationally expensive, poor scaling. **No simple `transform()`** for new data. Very sensitive to **perplexity**. Visualization only — **not** for feeding another model |
| **UMAP** | Nonlinear | Unsupervised | Manifold learning + topological data analysis: builds a **graph** of the data's structure, then optimizes a low-D layout to match | **Faster than t-SNE**, scales to larger data. **Preserves local + more of the global structure** than t-SNE. **Supports transforming new/unseen data** | Sensitive to hyperparameters (**n_neighbors, min_dist**). Prefer over t-SNE when speed matters or you need to embed new points |
| **LDA** (Linear Discriminant Analysis) | Linear | **SUPERVISED** | **Uses class labels** to find axes that **maximize separation between classes while minimizing variance within each class** | Optimizes for **class discrimination** rather than overall variance. **Can double as a classifier** | **Number of components is capped at (number of classes − 1)**. Assumes classes are roughly **Gaussian with equal covariance** — often violated |
| **Autoencoders** | **Nonlinear** | Unsupervised (self-supervised) | A neural net trained to **reconstruct its own input**, with a **bottleneck** hidden layer (the **latent space**) smaller than the input | Fully nonlinear (activation functions) → captures complex relationships PCA can't. The **encoder half can be reused** to transform new data. Variants: **denoising** (robust to noise), **variational/VAE** (generative, smooth latent space), **sparse** | **Needs more data and compute** than PCA. Use for large datasets, complex nonlinear structure, or when you want to also **generate/reconstruct** data |

**The one-line discriminator (if you learn only one thing here):**
**PCA = linear + unsupervised + variance. LDA = linear + SUPERVISED + class separability.
t-SNE/UMAP = nonlinear, visualization only (UMAP is faster and can transform new data).
Autoencoder = nonlinear, neural, and can reconstruct/generate.**

**Tools:** `scikit-learn` — PCA, TruncatedSVD, LDA · `openTSNE` / `sklearn.manifold` — t-SNE ·
`umap-learn` — UMAP.
**Key reading cited:** van der Maaten & Hinton, *"Visualizing Data using t-SNE"*, JMLR 2008 ·
McInnes, Healy & Melville, *"UMAP"*, arXiv:1802.03426, 2018.

---

## 4.13 The 5-step workflow that frames Lecture 6

| Step | Content | Techniques / code |
|---|---|---|
| **1. Data Collection** | Gathering raw data from multiple sources: **Databases (SQL, NoSQL), APIs (Twitter API, Weather API), Files (CSV, JSON, Excel), Streaming (IoT sensors, stock markets)**. Example datasets: **UCI Adult**, **Kaggle Titanic** | `pandas.read_csv("file.csv")` · `requests.get("api_url")` |
| **2. Data Preparation** | Data cleaning, feature engineering, data transformation | `df['Age'].fillna(df['Age'].median(), inplace=True)` · `df['Sex'] = df['Sex'].map({'male':0,'female':1})` |
| **3. EDA** | Understand patterns, distributions, relationships: **descriptive statistics (mean, median, std), visualization (histograms, scatter plots, heatmaps), clustering** | see Part 5 |
| **4. Machine Learning** | Build predictive models: **Classification (Logistic Regression, SVM), Regression (Linear Regression), Clustering (KMeans)** | scikit-learn |
| **5. Visualization & Deployment** | **Dashboards (Tableau, Power BI, Plotly Dash), Web Apps (Streamlit, Flask), Model APIs (FastAPI, TensorFlow Serving)** | — |

## 4.14 EDA tools worth naming
| Tool | Role |
|---|---|
| **pandas-profiling / ydata-profiling** | **One-line automated EDA reports** |
| **Sweetviz** | Fast **comparative** EDA reports |
| **matplotlib, seaborn, plotly** | Custom statistical visualizations |
| **missingno** | Missingness patterns / heatmaps |
