# 🎯 PAST PAPER ANALYSIS — this changes the whole revision strategy

## What the two past papers actually are

| | **Fall 2024** | **Fall 2025** |
|---|---|---|
| Date | Sep 21, 2024 | Sep 23, 2025 |
| Time | **1 hour** | **1 hour** |
| Marks | **60** | **50** |
| Questions | **3** | **3** |
| Pages | 4 | **9** (5 of them are a printed research paper) |
| Closed book | yes | yes, **no electronic device** |

> The two PDFs in `Past-Exams/` are the **same Fall-2025 paper**, two different students' copies
> (roll 0013 and roll 0020). The 4 WhatsApp images are the **Fall-2024** paper. So you have
> **two distinct papers**, not three.

---

## ⚠️ THE HEADLINE FINDING

### There is not a single theory/definition question in either paper.

No "define big data". No "list the stages of the DS life cycle". No "differentiate nominal and
ordinal". **Zero.** Both papers are **100% applied**: he puts a dirty table in front of you and asks
you to diagnose it, clean it, encode it, and write pandas code — plus (in 2025) read a research paper.

**Revision consequence:** the theory files (01–05) are now *background*, not the target. Your marks
come from **table diagnosis, encoding decisions, and pandas code**. Drill those.

### The tables REPEAT across years

The **online-retail table** (Customer_ID 1001–1020) and the **financial/Adult table** (15 rows) appear
**verbatim in both the 2024 and the 2025 paper**. He reuses his datasets. Learn these two tables cold
and you have pre-solved a large fraction of the paper.

---

## Fall 2024 paper — full structure (60 marks)

**Q1 [20 marks] — Online retail dataset (Table-1, IDs 1001–1020)**
- a. Identify the issues in the above dataset.
- b. Perform relevant encoding method to above dataset and provide the results.
- c. Perform an initial EDA.
- d. Apply the relevant data preprocessing method to handle the issues identified in part (a).
- e. Perform EDA again after preprocessing and compare the results of part (c).

**Q2 [20 marks] — Financial dataset (Table-2, the Adult-style 15-row table)**
- a. Apply the data preprocessing methods to clean this dataset.
- b. After preprocessing, perform the one-hot encoding of the dataset.
- c. Handle the issues (if you identify any in part (b)) using relevant technique.
- d. Apply an initial EDA (before any preprocessing) and again (after preprocessing) to compare.

**Q3 [20 marks] — IoT device dataset (Table-3, 20 rows)**
- a. Apply the one-hot encoding and label-encoding techniques to the above dataset and provide the results of both.
- b. **Perform the dimensionality reduction on the results of part (a).**
- c. Apply the relevant data-preprocessing methods to handle the data related issues (if any).

> Note the **"EDA before vs EDA after, compare"** pattern in Q1 and Q2 — that is *exactly* your
> Assignment-1 and Activity-1 structure. He examines what he assigns.

---

## Fall 2025 paper — full structure (50 marks)

**Q1 [15 marks] — Financial dataset (Table-1, same Adult-style table)**
- a. Apply the one-hot encoding and label-encoding techniques to the relevant column(s) and provide the results of both. **Code is not required.** (5)
- b. According to the results in (a), **which encoding technique is better and why?** (2)
- c. Identify the issues with the dataset. (3)
- d. Handle the issues identified in part (c) using relevant technique. **Code is not required.** (5)

**Q2 [15 marks] — Online retail dataset (Table-2, same retail table)**
- a. Identify the data related issues in the above dataset. (2)
- b. **Write Python code** to overcome the issues identified in part (a). (5)
- c. **Write a code** to compute the total purchase amount per country and sort descending. (3)
- d. For each customer, compute the total purchase amount. **Write a code** to identify the top 3 customers. (3)
- e. Find the number of purchases made per month across all years. **Write a code** to determine which month had the highest sales. (2)

**Q3 [20 marks] — RESEARCH PAPER, printed inside the exam (pages 6–9)**
Paper: *CLFSS — coarse-to-fine Contrastive Learning, Feature Selection and Synthesis for blood
pressure estimation from physiological vibration signals.*
- a. How does their proposed **CFCL** differ from existing frameworks? Why is a **hierarchical interval-based approach** particularly suitable for regression tasks such as blood pressure estimation?
- b. The authors designed a **Selection Layer (SL) with LASSO regularization**. How does this differ from traditional feature selection methods?
- c. What are the **strengths and weaknesses of using LASSO** for feature selection in physiological signals?
- d. The authors evaluated performance on public, private and synthetic datasets. **Why is testing under domain shift important?**
- e. Do you think metrics like **MAE and SD alone are sufficient** to assess clinical reliability?

> **40% of the 2025 paper was a research-paper comprehension question.** This is the single biggest
> change between 2024 and 2025, and it is the thing you are least prepared for.

---

## 📈 The trend, and what to expect tomorrow

| Element | 2024 | 2025 | Expect 2026 |
|---|---|---|---|
| Dirty-table diagnosis + fix | ✅ ×3 | ✅ ×2 | **✅ near-certain** |
| One-hot vs label encoding | ✅ ×2 | ✅ | **✅ near-certain** |
| "Which encoding is better and why" | — | ✅ | **likely** |
| Write pandas code (groupby/sort/top-n/dates) | — | ✅ | **likely — 2025 added this** |
| EDA before vs after, compare | ✅ ×2 | — | possible |
| Dimensionality reduction applied to encoded output | ✅ | — | possible |
| **Research paper comprehension** | — | **✅ 20/50** | **likely — prepare for it** |
| Pure theory/definition | ❌ | ❌ | **unlikely** |

**Most probable 2026 shape:** two applied dataset questions (one financial/Adult-style, one
retail/IoT-style) + one research-paper question. Budget your hour accordingly: ~15 min per dataset
question, ~25 min for the paper, 5 min to check.

---

# ★ MODEL ANSWER 1 — The FINANCIAL (Adult-style) table

**Columns:** Age, Workclass, Education, Marital_Status, Occupation, Relationship, Race, Sex,
Capital_Gain, Capital_Loss, Hours_per_Week, Native_Country, Income  (15 rows)

## (c/a) ISSUES — write them as a numbered list with the evidence beside each

| # | Issue | Evidence from the table |
|---|---|---|
| 1 | **Disguised missing values — `?` used instead of NaN** | `Workclass` row 6; `Occupation` rows 1 and 6; `Relationship` row 12; `Native_Country` row 14. `df.isna()` reports **zero** until these are converted. |
| 2 | **Genuine NaN missing values** | `Age` = NaN (the Self-emp-inc row); `Sex` = NaN (the Assoc-acdm row) |
| 3 | **Inconsistent categories in `Sex`** | **Male, M, Female, F** — four labels for two categories, plus a NaN |
| 4 | **Severely inconsistent `Native_Country`** ← *the headline issue* | **US, United-states, United-State, USA, United-States, America, United-States-of-America** = **7 spellings of one country**, plus `?` |
| 5 | **Zero-variance / constant feature** | **`Capital_Loss` is 0 for every single row** → carries no information → drop it |
| 6 | **Extremely skewed / sparse feature** | `Capital_Gain` is 0 for 12 of 15 rows; non-zero only 2174, 5178, 14084 → sparse, heavy right skew → consider log1p or binning to a flag |
| 7 | **Outliers in `Hours_per_Week`** | **80** and **67** at the top, **16** at the bottom vs a mode of 40 → *unusual but possible — outliers, not errors.* Do **not** delete |
| 8 | **`Education` is ORDINAL but stored as unordered text** | 9th < 11th < HS-grad < Some-college < Assoc-acdm/Assoc-voc < Bachelors < Masters — one-hot would destroy that order |
| 9 | **Class imbalance in the target `Income`** | **9 × ≤50K vs 6 × >50K** (60/40 here; 76/24 in the full Adult dataset) |
| 10 | **Wrong dtypes** | `Age` becomes float (or object) because of the NaN; every categorical is `object` |
| 11 | **Redundancy between `Marital_Status` and `Relationship`** | Married-civ-spouse ↔ Husband/Wife encode the same fact twice → redundant features |

> Careful — do **not** call *Married-civ-spouse / Married-spouse-absent / Never-married / Divorced*
> an inconsistency. They are genuinely different categories. Over-calling loses marks.

## (a) ENCODING — one-hot and label, applied to the RELEVANT columns

State the decision rule first, then show the result. That framing earns the marks.

**Step 1 — classify each column by measurement level:**

| Column | Type | Correct encoding |
|---|---|---|
| Age, Capital_Gain, Capital_Loss, Hours_per_Week | Numeric (ratio) | **none** — scale, don't encode |
| Sex | **Binary nominal** | **Label / binary** (Male=1, Female=0) — 1 column is enough |
| Income (target) | **Binary nominal (target)** | **Label** (≤50K=0, >50K=1) |
| Education | **Ordinal** | **Ordinal encoding** preserving rank (9th=0 … Masters=6) |
| Workclass, Marital_Status, Occupation, Relationship, Race, Native_Country | **Nominal** | **One-hot** |

**Step 2 — show a worked example of each** (the examiner wants to see you can actually do it):

*Label encoding on `Sex`:* Male → 1, Female → 0
*Label encoding on `Income`:* `<=50K` → 0, `>50K` → 1
*Ordinal encoding on `Education`:* 9th→0, 11th→1, HS-grad→2, Some-college→3, Assoc-voc→4, Assoc-acdm→5, Bachelors→6, Masters→7

*One-hot on `Race` (after cleaning):*

| Race | Race_White | Race_Black | Race_Asian-Pac-Islander |
|---|---|---|---|
| White | 1 | 0 | 0 |
| Black | 0 | 1 | 0 |
| Asian-Pac-Islander | 0 | 0 | 1 |

*One-hot on `Workclass`:* `Workclass_Private`, `Workclass_State-gov`, `Workclass_Self-emp-not-inc`,
`Workclass_Self-emp-inc` (+ `Workclass_Unknown` if you keep `?` as its own category).

*Label encoding on `Workclass` (what you'd get, and why it's wrong):*
Private→0, Self-emp-inc→1, Self-emp-not-inc→2, State-gov→3

## (b) WHICH ENCODING IS BETTER AND WHY — [2 marks, high-value, answer like this]

> **Neither is universally better — the right choice depends on the measurement level of the column.**
>
> - For the **nominal** columns (Workclass, Marital_Status, Occupation, Relationship, Race,
>   Native_Country), **one-hot encoding is better.** Label encoding assigns integers (Private=0,
>   Self-emp-inc=1, State-gov=3) which **imposes a false order and a false notion of distance** — the
>   model would infer that State-gov > Private and that Private is "closer" to Self-emp-inc than to
>   State-gov. That is meaningless, and it corrupts every distance-based (kNN, k-means, SVM) and
>   linear model. One-hot keeps the categories **equidistant and independent**.
> - For the **ordinal** column **Education**, **label/ordinal encoding is better**, because the order
>   9th < 11th < HS-grad < … < Masters is **real information**; one-hot would throw it away and add
>   8 columns for nothing.
> - For the **binary** columns (**Sex, and the Income target**), **label encoding is better** — one
>   column carries all the information, while one-hot produces a redundant second column that is a
>   perfect linear function of the first (**the dummy-variable trap / perfect multicollinearity**).
> - **Cost of one-hot:** it **increases dimensionality** — Native_Country alone would produce 7+
>   columns from 15 rows, feeding the **curse of dimensionality**. Mitigate with `drop_first=True`,
>   by grouping rare categories into "Other", or by using target/hashing encoding for high cardinality.
>
> **Verdict for this dataset:** one-hot for the nominal columns, ordinal for Education, label for
> Sex and Income — *after* the categories have been standardized, because encoding a dirty column
> (7 spellings of USA) would create 7 useless dummy columns.

> 🔑 **That last sentence is the money line: you must CLEAN BEFORE YOU ENCODE.** In 2024 Q2 he made
> the order explicit ("after preprocessing, perform the one-hot encoding"). Say it out loud.

## (d) HANDLING THE ISSUES — the Problem → Action → Justification table

| Issue | Action | Justification |
|---|---|---|
| `?` placeholders | `df.replace('?', np.nan)` **first** | Until converted, every missing-value count is wrong |
| Missing Workclass / Occupation / Native_Country | Impute with the **mode**, or keep an explicit **"Unknown"** category | Mode is the only central tendency for nominal data; "Unknown" preserves the fact that it was not reported, which may itself be informative |
| Missing Age | Impute with the **median** | Robust to the skew/outliers already present; the mean would be dragged |
| Missing Sex | Impute with **mode**, or drop the single row | Only 1 of 15 rows; either is defensible if you state which |
| `Sex` inconsistency (Male/M/Female/F) | `.str.strip().str.upper().str[0]` → {M, F}, then label encode | Collapses 4 labels to the 2 real categories |
| `Native_Country` — 7 spellings of USA | Build an explicit **canonical mapping dictionary** → all 7 map to `"United-States"` | This is **entity resolution / schema harmonization**. Without it, one-hot creates 7 spurious columns and splits the same country's signal |
| `Capital_Loss` constant 0 | **Drop the column** | Zero variance → no information → pure dimensionality cost |
| `Capital_Gain` sparse and skewed | `np.log1p()`, or binarize to `has_capital_gain` | Tames the skew; `log1p` handles the zeros safely |
| Hours_per_Week outliers (80, 67, 16) | **Retain** (or cap/winsorize) — **do not delete** | They are **unusual but possible**, i.e. outliers, not errors. Deleting real extremes biases the population represented |
| `Education` ordinal | **Ordinal encode**, preserving rank | Preserves genuine order information |
| Nominal columns | **One-hot** *after* standardizing | Avoids inventing a false ordering |
| Target `Income` | **Label encode** (≤50K=0, >50K=1) | Binary target; 1 column suffices |
| Class imbalance 9:6 | Note it; use **stratified split**, and evaluate with **precision/recall/F1**, not accuracy | Accuracy is misleading on imbalanced targets |
| Marital_Status ↔ Relationship redundancy | Keep one, or note the redundancy | Redundant features duplicate information (the course's own definition) |
| Numeric columns | **StandardScaler** | Puts features on a comparable scale for distance-based and gradient-based algorithms |

---

# ★ MODEL ANSWER 2 — The ONLINE RETAIL table (this is the one he asks code on)

**Columns:** Customer_ID, Name, Age, Gender, Income, Product_Category, Purchase_Amount, Country, Sales_Date (20 rows, IDs 1001–1020)

## (a) ISSUES — verified counts, quote these numbers

| # | Issue | Verified evidence |
|---|---|---|
| 1 | **Missing values** | `Age` = 3 NaN · `Income` = 3 NaN · `Name` = 1 NaN (ID 1007) · `Country` = 1 NaN (ID 1014) |
| 2 | **Inconsistent `Gender`** | **5 raw labels → 2 real**: `{F, Female, M, Male, female}` |
| 3 | **Inconsistent `Country`** | **9 raw labels → 3 real**: `{CANADA, Canada, UK, USA, United States, United_Kingdom, Usa, uk, usa}` → {USA, UK, Canada} |
| 4 | **Inconsistent `Product_Category`** | **7 raw labels → 3 real**: `{Clothing, clothing, Electronics, electronics, Elec, Furniture, furniture}` → {clothing, electronics, furniture}. Note **`Elec` is an abbreviation**, not just a case difference |
| 5 | **Inconsistent `Name` capitalisation** | **11 raw spellings → 9 real customers**: `john doe / JANE SMITH / LUCY LEWIS / lucy lewis / Robert Brown / ROBERT BROWN / mary ann …` |
| 6 | **Negative `Purchase_Amount`** | **−50** for ID 1007 → **impossible, an ERROR not an outlier** |
| 7 | **Mixed date formats** | 19 rows are `M/D/YYYY` (3/10/2022) but ID 1008 is **`15-08-2022` (`D-M-YYYY`)** → parsing fails or silently mis-parses |
| 8 | **Duplicate records** | **4 duplicate rows** once you ignore Customer_ID and Sales_Date (e.g. 1005≡1015 Lucy Lewis; 1003≡1009 Robert Brown; 1001≡1020 john doe) |
| 9 | **`Customer_ID` is NOT a customer identifier** ← *the trap* | The same person has several IDs: john doe = 1001, 1004, 1020; Lucy Lewis = 1005, 1010, 1015; Bob Johnson = 1013, 1019. It is effectively a **transaction ID** |
| 10 | **Value conflicts for the same entity** | Bob Johnson: `Income` = NaN (1013) vs 80000 (1019). Jane Smith: 80000 (1002) vs NaN (1017) → impute **within** the customer, not from the global mean |
| 11 | **Wrong dtypes** | `Age`/`Income` float because of NaN; `Sales_Date` is a **string**, not datetime |

> 🔑 **Issue 9 is the one almost everyone misses and it changes part (d).** "For each **customer**"
> means group by the **cleaned Name**, not by `Customer_ID`. Grouping by `Customer_ID` gives a
> meaningless three-way tie at 1200. Say this explicitly in your answer — it's a distinguishing point.

## (b) PYTHON CODE to fix the issues — [5 marks]

```python
import pandas as pd, numpy as np

df = pd.read_csv("retail.csv")

# 1. Standardise text: strip whitespace, fix capitalisation
df["Name"]             = df["Name"].str.strip().str.title()
df["Gender"]           = df["Gender"].str.strip().str.upper().str[0]          # {M, F}
df["Product_Category"] = (df["Product_Category"].str.strip().str.lower()
                            .replace({"elec": "electronics"}))                 # fix abbreviation

# 2. Canonical mapping for Country (entity resolution)
country_map = {"usa": "USA", "united states": "USA",
               "uk": "UK",  "united_kingdom": "UK",
               "canada": "Canada"}
df["Country"] = df["Country"].str.strip().str.lower().map(country_map)

# 3. Errors -> NaN  (negative purchase amount is impossible)
df.loc[df["Purchase_Amount"] < 0, "Purchase_Amount"] = np.nan

# 4. Mixed date formats -> datetime
df["Sales_Date"] = pd.to_datetime(df["Sales_Date"], format="mixed", dayfirst=False)

# 5. Missing values
df["Age"]    = df.groupby("Name")["Age"].transform(lambda s: s.fillna(s.median()))
df["Income"] = df.groupby("Name")["Income"].transform(lambda s: s.fillna(s.median()))
df["Age"]    = df["Age"].fillna(df["Age"].median())        # fallback: global median
df["Income"] = df["Income"].fillna(df["Income"].median())
df["Country"] = df["Country"].fillna(df["Country"].mode()[0])
df["Purchase_Amount"] = df["Purchase_Amount"].fillna(df["Purchase_Amount"].median())
df = df.dropna(subset=["Name"])                             # cannot recover an unknown customer

# 6. Duplicates — dedupe on the BUSINESS key, not the surrogate Customer_ID
df = df.drop_duplicates(
        subset=["Name", "Age", "Gender", "Product_Category",
                "Purchase_Amount", "Country", "Sales_Date"])

# 7. Dtypes
df["Age"] = df["Age"].astype(int)
```

> **Why `groupby("Name").transform(...)` for Age/Income:** the same customer appears several times,
> so the most defensible imputation is **that customer's own value from another row** (Robert Brown's
> age, Bob Johnson's income) rather than a global average. Saying this earns the justification marks.

## (c) Total purchase amount per country, sorted descending — [3 marks]

```python
df.groupby("Country")["Purchase_Amount"].sum().sort_values(ascending=False)
```
**Verified result:** `USA 8950 · UK 2650 · Canada 2300`

## (d) Total purchase per customer, top 3 — [3 marks]

```python
customer_totals = df.groupby("Name")["Purchase_Amount"].sum()
top3 = customer_totals.nlargest(3)          # or .sort_values(ascending=False).head(3)
```
**Verified result:** `Lucy Lewis 3000 · Robert Brown 2650 · Jane Smith 2400`

> Add the sentence: *"I group by the cleaned `Name` rather than `Customer_ID`, because the same
> customer appears under multiple Customer_IDs — that column is a transaction ID, not a customer key."*

## (e) Purchases per month across all years; which month had the highest sales — [2 marks]

```python
df["Sales_Date"] = pd.to_datetime(df["Sales_Date"], format="mixed")

purchases_per_month = df.groupby(df["Sales_Date"].dt.month).size()      # COUNT of purchases
sales_per_month     = df.groupby(df["Sales_Date"].dt.month)["Purchase_Amount"].sum()  # REVENUE

print(purchases_per_month)
print("Month with most purchases:", purchases_per_month.idxmax())
print("Month with highest sales :", sales_per_month.idxmax())
```
**Verified result:** counts `{Mar 1, May 1, Jun 1, Jul 4, Aug 1, Sep 1, Oct 5, Nov 5, Dec 1}` →
**October and November tie at 5 purchases**; by **revenue, November is highest (4250)** vs October (4050).

> The question says "number of purchases ... which month had the **highest sales**" — **ambiguous**.
> **Compute both and say so in one line.** That converts an ambiguity into a mark instead of a risk.
> *(Assumes the −50 error was set to NaN and the `15-08-2022` row parsed as 15 August.)*

---

# ★ MODEL ANSWER 3 — The IoT table (Fall-2024 Q3)

**Columns:** Device_ID, Temperature (°C), Humidity (%), Battery_Level (%), Motion_Detected,
Timestamp, Location, Device_Type, Network_Signal_Strength (dBm), Data_Usage (MB)  (20 rows)

## Issues
| # | Issue | Evidence |
|---|---|---|
| 1 | **`?` as a disguised missing value, scattered across many columns** | Temperature (rows 3, 11) · Humidity (rows 2, 13) · Battery_Level (rows 7, 15) · Network_Signal (rows 4, 20) · Data_Usage (rows 3, 4) |
| 2 | **Inconsistent boolean representation in `Motion_Detected`** | Mixed **`Yes` / `No` / `0`** — three tokens for two states |
| 3 | **Impossible value / unit inconsistency** | **Temperature = 98.26 °C** (row 19). A room is not 98 °C — this is **98.26 °F ≈ 36.8 °C**, i.e. a unit mix-up, or a sensor fault. **An error, not an outlier** |
| 4 | **Inconsistent category labels in `Device_Type`** | **`Thermostat` vs `Smart Thermostat`** — almost certainly the same device type written two ways |
| 5 | **Outliers in `Data_Usage`** | **22 MB** (row 12, Smart TV) and **39 MB** (row 20) vs ~0.7–4 MB elsewhere. *For a Smart TV 22 MB is plausible → outlier, not error.* Judge per Device_Type |
| 6 | **`Timestamp` stored as text** | `9/1/2024 10:15` → must become datetime to extract hour/minute features |
| 7 | **`Device_ID` is an identifier, not a feature** | Irrelevant feature — drop before modelling (cf. "student ID for predicting GPA") |
| 8 | **Negative values in `Network_Signal_Strength` are NOT errors** | dBm is **legitimately negative** (−40 to −65). **Do not "fix" them** — a trap |

## (a) Encoding
- **One-hot:** `Location` (Living Room, Bedroom, Kitchen, Garden → 4 columns) and `Device_Type`
  (Thermostat, Smart Bulb, Smart Speaker, Smart TV, Security Camera → 5 columns **after** merging
  "Smart Thermostat" into "Thermostat")
- **Label/binary:** `Motion_Detected` → Yes=1, No=0 (after normalising the stray `0` to `No`)
- **None:** all the numeric sensor readings — scale them instead

## (b) Dimensionality reduction on the encoded result — *the interesting part*
One-hot turned 2 columns into ~9, so with ~8 numeric columns you now have ~17 features for 20 rows —
**more features than you have comfortable rows: the curse of dimensionality in miniature.** Options:
1. **`drop_first=True`** on the dummies — removes the redundant column per variable (the **dummy-variable trap**), 9 → 7.
2. **Group rare categories** into "Other" before encoding.
3. **PCA** on the scaled numeric block — projects onto orthogonal directions of maximum variance.
   *State the caveat:* PCA on 0/1 dummy columns is questionable because they are not continuous;
   prefer PCA on the numeric sensor block, or MCA/one of the categorical-aware methods.
4. **Feature subset selection** — drop `Device_ID` (irrelevant) and any zero-variance column.
5. Because **n = 20 rows**, note that **t-SNE/UMAP would be unreliable** at this sample size.

> Mentioning "17 features for 20 rows" and the dummy-variable trap is what turns a 2/5 into a 5/5 here.

---

# ★ Q3 — THE RESEARCH PAPER QUESTION (20 marks, 40% of the paper)

## The crucial fact: the paper is PRINTED IN THE EXAM

In 2025 the full paper occupied pages 6–9 of the question paper. **You do not need to have read it
in advance.** What you need is a **reading protocol** and a set of **transferable answers**, because
his five sub-questions follow the same five shapes every time:

| Shape | 2025 wording | What he's really testing |
|---|---|---|
| **1. Novelty** | "How does the proposed X differ from existing frameworks? Why is approach Y suitable for task Z?" | Can you find the contribution paragraph and contrast it with Related Works? |
| **2. Mechanism** | "The authors designed a Selection Layer with LASSO. How does this differ from traditional feature selection?" | Do you know the **Filter / Wrapper / Embedded** taxonomy? ← *straight out of Lecture 3* |
| **3. Critique** | "Strengths and weaknesses of using LASSO for feature selection in physiological signals?" | Can you criticise a method, not just describe it? |
| **4. Evaluation design** | "Why is testing under domain shift important?" | Do you understand generalization vs memorization? |
| **5. Metric adequacy** | "Are MAE and SD alone sufficient to assess clinical reliability?" | Do you know that a single metric hides failure modes? ← *Lecture 2 stage 8* |

## Your 5-minute reading protocol (do this before writing anything)

1. **Abstract** → the problem, the proposed method's name, the headline result.
2. **Last paragraph of the Introduction** → the bulleted **"contributions"** list. This answers shape 1.
3. **Related Works** → what existed before. The contrast for shape 1.
4. **Methodology headings only** → the module names and what each does. This answers shape 2.
5. **Experiments → datasets + metrics + results table** → answers shapes 4 and 5.
6. **Conclusion → "In the future, we plan to…"** → the authors' own admission of limitations. Free marks for shape 3.

## Pre-built answers you can adapt to almost any paper

**Shape 2 — "how does this differ from traditional feature selection?"** (the Lecture-3 taxonomy is the answer)
> Traditional feature selection falls into three families: **Filter** methods score features
> independently of the model (correlation, chi-square, mutual information) — fast but they **ignore
> feature interactions**; **Wrapper** methods (RFE) use the learning algorithm as a black box and
> **do account for interactions but are computationally very costly**; **Embedded** methods (LASSO,
> Ridge, tree importances) perform selection **as part of model training**.
> A **Selection Layer with LASSO regularization** is an **embedded** method taken one step further:
> it is a **one-to-one connected layer without bias**, so each neuron is exclusively linked to a
> single input feature, and the **L1 penalty drives the weights of irrelevant features to exactly
> zero** during gradient descent — the gradient of the L1 norm stays constant at ±1, so the
> regularization pressure is maintained at a consistent intensity until the weight reaches zero.
> The key difference: selection is **learned jointly and end-to-end with the downstream task**
> rather than performed as a **separate pre-processing step**, so the features are selected *for the
> objective that actually matters* instead of for a generic proxy score.

**Shape 3 — strengths and weaknesses of LASSO**
> **Strengths:** performs selection and estimation **simultaneously**; produces **exact zeros**, so
> the result is a genuinely **sparse, interpretable** feature subset; **embedded**, so no expensive
> outer search like RFE; **regularizes**, which reduces overfitting when features ≫ samples — common
> in physiological signals where hundreds of TSFEL features are extracted from a short window.
> **Weaknesses:** with **highly correlated features it arbitrarily picks one and zeroes the rest**,
> which is unstable — and physiological features (peak distances, heart-rate derivatives) *are*
> strongly correlated, so a small change in the data can change the selected set; it assumes a
> **linear** feature–target relationship, so it can miss **nonlinear** dependencies, which is why the
> authors add an **MLP synthesizer** after the Selection Layer; the **λ penalty needs tuning** by
> cross-validation; and it can **over-shrink** genuinely useful coefficients, biasing them toward zero.

**Shape 4 — why testing under domain shift matters**
> **Domain shift** means the data distribution at deployment differs from the training distribution —
> a different sensor, a different hospital, different demographics, a different SBP range. A model
> that is only validated on an i.i.d. split of one dataset can score well simply by exploiting
> **domain-specific artifacts**, which is a form of **memorization rather than generalization**.
> Testing on a **public, a private/self-collected, and a synthetic label-shifted** dataset probes
> exactly this: in the paper, the baselines "**all suffered a significant decline**" on the synthetic
> label-shifted SBP data while the proposed framework held up, which is evidence that its
> representations are **domain-invariant** rather than memorized. It also connects to the course's
> **Monitoring & Maintenance** stage: real-world data changes after deployment (**drift**), so a
> model that cannot survive a domain shift in the lab will fail in production. Finally, in a
> **clinical** setting this is a safety issue, not just an accuracy issue.

**Shape 5 — are MAE and SD alone sufficient?**
> **No.** They are necessary but not sufficient. Reasons:
> (1) **MAE is an average — it hides the tail.** A model can have an excellent mean error while
> catastrophically mis-estimating the hypertensive patients who matter most clinically.
> (2) **Neither reveals bias direction or calibration** — a consistent −5 mmHg offset and random
> ±5 mmHg noise can give the same MAE but have very different clinical meaning. Bland–Altman
> analysis, mean error (ME) and limits of agreement are the standard here.
> (3) **No per-subgroup breakdown** — performance may differ by age, sex, BMI or skin tone;
> aggregate metrics conceal **subgroup failure**, which the course flags in the Evaluation stage
> ("evaluate robustness, fairness, leakage and generalization — not only headline accuracy").
> (4) **No uncertainty estimate** — clinical use needs to know *when the model does not know*.
> (5) **Standards demand more:** the paper itself cites **FDA / ISO 81060-2:2018(E)**, which requires
> a mean error within **±5 mmHg** and an SD no greater than **8 mmHg** — a pass/fail protocol over a
> prescribed subject population, not just a reported average.
> **What to add:** Bland–Altman limits of agreement, error distribution and worst-case percentiles,
> per-subgroup and per-BP-range breakdowns, correlation/calibration plots, prospective clinical
> validation, and test–retest reliability.

**Shape 1 — novelty** (template; fill from the paper's contribution bullets)
> Existing methods do A, which has limitation B. The authors propose **X**, which differs in that it
> **[mechanism]**, thereby addressing B. It is suitable for **[task]** because **[property of the
> task] matches [property of the mechanism]**.
> *2025 instance:* most contrastive-learning approaches are **designed for classification**, so they
> emphasise **global** semantic features and overlook the **local** detail needed to distinguish
> closely related labels — which is fatal for **regression**, where the target is continuous and
> nearby values must be told apart. **CFCL** instead performs **hierarchical binary classification
> into progressively narrower intervals**, applying a contrastive loss at **each level of
> granularity** (same interval = positive pair, different interval = negative pair), with the final
> interval constrained to **≤10 mmHg**. That coarse-to-fine structure mirrors the **ordered,
> continuous nature of blood pressure**: coarse levels capture global structure, fine levels capture
> the local resolution a regressor needs. It is also fused with DSP-derived synthesized features,
> combining the rich semantics of learned representations with the stability of hand-crafted ones.

## The three papers HE shared — read the abstracts tonight
If he uses one of these instead of an unseen paper, you'll be ahead:

| Paper | Core idea | Likely question angle |
|---|---|---|
| **Combining structured and unstructured data for predictive models: a deep learning approach** (Zhang et al., BMC 2020) | 2 **multi-modal fusion** architectures combining **sequential clinical notes** (document embeddings + CNN/LSTM) with **structured** time-series + one-hot static data, on **MIMIC-III**; tasks: in-hospital mortality, 30-day readmission, long length-of-stay | Structured vs unstructured data; why fusion beats either alone; data fusion; patient representation learning |
| **Multi-Class Imbalance Classification Based on Data Distribution and Adaptive Weights** (IEEE TKDE 2024) | AdaBoost for multi-class imbalance; adds a **density-based factor** so samples in the *same* class get *different* weights, plus an **adaptive weighting** mechanism using up-to-date training status | Class imbalance (your Adult target is imbalanced!); why accuracy fails; ensembles; static vs adaptive weights |
| **Optimal Feature Selection-Based Medical Image Classification Using DL in IoMT** (IEEE Access 2020) | **Opposition-based Crow Search (OCS)** picks optimal multi-texture/grey-level features before a DL classifier; lung cancer, brain, Alzheimer's; **95.22% accuracy, 86.45% sensitivity, 100% specificity** | Feature selection (Filter/Wrapper/Embedded), preprocessing → selection → classification pipeline, metric adequacy |

> Thematically, the 2025 exam paper (feature selection + LASSO + medical signals) is **closest to the
> IoMT feature-selection paper**. If you only skim one, skim that one — but the Lecture-3
> Filter/Wrapper/Embedded taxonomy is the reusable weapon for all of them.

---

# 📋 EXAM-DAY TACTICS

1. **Clean before you encode.** Say it explicitly. He sequenced it that way in 2024 Q2.
2. **Always structure issue-lists as a table: Issue → Evidence → Action → Justification.** That is
   the Activity-1 format he taught you; it is what he is marking against.
3. **Never write "remove the outliers."** Write: *"these are outliers, not errors — I retain/cap
   them because they are unusual but possible."* Only delete things that are **impossible**.
4. **Count things and quote the count.** "9 spellings of 3 countries", "4 duplicate rows",
   "Capital_Loss is 0 in all 15 rows". Specific numbers read as evidence; vague claims don't.
5. **When a question is ambiguous, answer both readings in one line** (the "highest sales" =
   count-or-revenue case). That's a mark gained, not a risk taken.
6. **"Code is not required" means show the RESULT** — draw the encoded table. Don't waste time
   writing pandas when he asked for the output table.
7. **Watch the clock: 1 hour, 3 questions.** In 2025 the paper question was 40% of the marks. Don't
   spend 40 minutes perfecting a cleaning table and leave the paper unanswered.
8. **For the paper question, read the Abstract + contributions + conclusion FIRST**, write the
   five answers, and only dip into the methodology for the specific mechanism he names.
