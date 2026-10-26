# Viva preparation guide - Assignment #01 (Pre-processing and EDA)

Taha Abbas Naqvi, 261-8004, MS (DS), Section A

This file teaches the whole assignment from zero: the vocabulary, the tools, every
step you performed, why that step and not another one, and the questions an examiner
is likely to ask. Read Parts 1 to 3 first if the basics are shaky. If you are short on
time, read Part 0, Part 9 and Part 8 in that order.

---

## PART 0 - The 60 second answer

If the examiner opens with "tell me what you did", say this:

> I was given two datasets. The attached one has 5050 rows and 20 columns, and the
> Census-Income KDD dataset has 199,523 rows and 42 columns. I ran EDA on both before
> touching anything, wrote down the problems I could prove from that EDA, cleaned both,
> and then ran the same EDA again to show the problems were gone.
>
> The attached dataset was dirty in its values: the string "unknown" was used instead of
> blanks, which made two numeric columns load as text, plus 1509 empty cells, 49 duplicate
> rows, one country written three different ways, and impossible values like 391 hours in a
> week and a BMI of 204.
>
> The census dataset had clean values but a hard structure: no header row, missing values
> written as "?", 46,627 duplicate records, a survey weight that is not allowed to be a
> feature, nominal codes stored as integers, and a 94 to 6 class imbalance.
>
> After cleaning, both datasets have zero missing cells, zero duplicates, zero text columns,
> and all features on a comparable scale.

That is the whole assignment in one paragraph. Everything below is the backup.

---

## PART 1 - Vocabulary from zero

### 1.1 The table

| Term | Meaning |
|---|---|
| Dataset | A table. Rows are records, columns are variables. |
| Row | One observation, one record, one instance. In these datasets, one person. |
| Column | One variable, one attribute, one feature. |
| Feature | An input column, something you use to predict with. |
| Target / label | The column you would predict. `disease_risk` in dataset 2, `income_label` in dataset 1. |
| Shape | (rows, columns). `df.shape` gives it. |
| Index | Row labels. Pandas gives 0, 1, 2, ... by default. |

### 1.2 Types of data

Two big families, and the split decides how you clean and encode:

**Numerical**
- Discrete: counts, whole numbers. `weeks_worked`, `num_persons_worked`.
- Continuous: any value in a range. `bmi`, `annual_income`.

**Categorical**
- Nominal: named groups with no order. `gender`, `occupation`, `native_country`.
  You cannot say Sales > Tech-support.
- Ordinal: named groups with a real order. `education` (High School < Bachelors <
  Masters < PhD), `physical_activity` (low < medium < high).
  Order exists but the gaps are not measurable, PhD minus Masters is not a number.

This distinction is the single most important idea in the encoding part of the
assignment. Nominal gets one-hot, ordinal gets integer codes.

### 1.3 Data types in pandas (dtype)

`dtype` is how pandas stores a column in memory.

| dtype | What it holds |
|---|---|
| `int64` | Whole numbers |
| `float64` | Decimal numbers, and this is the only type that can hold NaN |
| `object` / `str` | Text, or a mix pandas could not parse |
| `bool` | True/False |
| `category` | Text stored as codes plus a lookup, saves memory |

Key fact for your viva: **if even one value in a numeric column is text, pandas loads
the whole column as text.** That is exactly what the string "unknown" did to
`annual_income` and `credit_score`. The proof is `df.info()`, where those two columns
show up as `str` while every other numeric column shows `int64` or `float64`. They also
disappear from `df.describe()`, because describe only summarises numeric columns.

### 1.4 Missing data

| Term | Meaning |
|---|---|
| NaN | "Not a Number". The marker pandas uses for a missing value. |
| Sentinel value | A normal looking value used to mean "missing": "unknown", "?", "NA", -999, 0. |
| `isna()` | True/False per cell, whether it is NaN. |
| `isna().sum()` | Count of missing values per column. |

A sentinel is more dangerous than a blank, because pandas does not know it means
"missing". It counts it as a real category, it changes the dtype, and it quietly
survives into the model. Both your datasets had sentinels: "unknown" in dataset 2 and
"?" in dataset 1.

### 1.5 Describing a distribution

A **distribution** is just the shape you get when you plot how often each value occurs.

**Centre**
- Mean: the average. Sensitive to outliers. One salary of 2.29 million drags it up.
- Median: the middle value when sorted. Robust, an outlier only moves it one position.
- Mode: the most frequent value. The only one that works for text columns.

**Spread**
- Range: max minus min.
- Standard deviation (std): typical distance from the mean. Same units as the data.
- Variance: std squared.
- Quartiles: Q1 is the 25th percentile, Q2 the median, Q3 the 75th percentile.
- IQR: Q3 minus Q1, the width of the middle 50 percent of the data. Robust to outliers.

**Shape**
- Skewness: how lopsided the distribution is.
  - 0 means symmetric.
  - Positive means a long right tail (a few very large values). Income is the classic case.
  - Negative means a long left tail.
  - Rule of thumb: |skew| under 0.5 is fine, 0.5 to 1 is moderate, above 1 is substantial.
  - Your `annual_income` had skew 16.09, which is extreme.
- Kurtosis: how heavy the tails are, how much of the data sits far from the centre.
  Pandas reports excess kurtosis, so a normal distribution gives 0. Your census
  `dividends` had 1090 before the log transform, which is a warning sign on its own.

### 1.6 Outliers

An outlier is a value far away from the rest of the data. Three different things can
cause one, and they need different treatment:

1. A data entry error: BMI of 204, 391 hours in a week. These are wrong and must go.
2. A genuine extreme value: a real CEO salary. It is true data, but it can still
   dominate the mean and the scaler.
3. A different population accidentally mixed in: children inside an adult income survey.

Detection rule used in your notebook (Tukey's rule, the one behind every boxplot):

```
lower fence = Q1 - 1.5 * IQR
upper fence = Q3 + 1.5 * IQR
anything outside the fences is flagged as an outlier
```

The other common rule is the z-score: flag anything more than 3 standard deviations from
the mean. You did not use it, and you have a reason: the z-score uses the mean and the
std, both of which are themselves distorted by the outliers you are trying to find. The
IQR rule uses quartiles, which are not.

### 1.7 Correlation

Pearson correlation measures **linear** association between two numeric columns.

- Runs from -1 to +1. 0 means no linear relationship.
- +1 means as one goes up the other goes up in a straight line.
- It only sees straight lines. A perfect U shaped relationship gives roughly 0.
- Correlation is not causation. Education correlating with income does not prove
  education causes income.

### 1.8 EDA and preprocessing

- **EDA (Exploratory Data Analysis)**: looking at the data before modelling. Shapes,
  types, missing values, summary statistics, distributions, relationships. Its purpose
  is to find out what you actually have.
- **Preprocessing**: making the data fit to use. Fixing errors, filling gaps, removing
  duplicates, encoding text into numbers, putting features on a comparable scale.

In this assignment the two are interleaved on purpose: EDA finds the problems, the
preprocessing fixes them, the second EDA proves the fix worked.

---

## PART 2 - The tools you used

### 2.1 The four libraries

| Library | Role in your notebook |
|---|---|
| **numpy** | Numeric arrays, `np.nan` for missing values, `np.log1p` for the log transform. Pandas is built on top of it. |
| **pandas** | The DataFrame. Everything to do with loading, inspecting, cleaning and reshaping the tables. |
| **matplotlib** | The base plotting engine. Figures, axes, subplots. |
| **seaborn** | A layer on top of matplotlib with better defaults and statistical plots (histplot, boxplot, countplot, heatmap). |
| **scikit-learn** | The machine learning library. You used only its preprocessing tools: SimpleImputer, StandardScaler, OrdinalEncoder. |

### 2.2 The pandas functions in your notebook

| Call | What it does |
|---|---|
| `pd.read_csv(path)` | Loads a csv into a DataFrame. |
| `df.head(n)` | First n rows. |
| `df.info()` | Column names, non-null counts, dtypes, memory. This is where hidden sentinels show up. |
| `df.describe()` | count, mean, std, min, 25%, 50%, 75%, max for numeric columns only. |
| `df.isna().sum()` | Missing values per column. |
| `df.duplicated().sum()` | How many rows are exact copies of an earlier row. |
| `df.drop_duplicates(keep="first")` | Removes them. |
| `df.replace(a, b)` | Value substitution, used for sentinels and country labels. |
| `pd.to_numeric(s, errors="coerce")` | Converts text to numbers; anything unconvertible becomes NaN. |
| `df[col].value_counts()` | Frequency of each level. `dropna=False` also counts NaN. |
| `df[col].nunique()` | Number of distinct levels, that is cardinality. |
| `df[col].quantile([0.01, 0.99])` | Percentiles, used for the capping limits. |
| `df[col].clip(lo, hi)` | Pulls values outside the range back to the limits. This is winsorising. |
| `df[col].skew()` / `.kurtosis()` | Shape statistics. |
| `df.corr()` | Correlation matrix. |
| `pd.get_dummies(df, columns=[...], drop_first=True)` | One-hot encoding. |
| `df.groupby(col)[other].mean()` | Split into groups and aggregate. |
| `s.map(dict)` | Element by element replacement, used for ordinal codes and frequency encoding. |
| `s.astype(t)` | Type conversion. |
| `s.str.strip()` | Removes leading and trailing spaces from text. |

### 2.3 The scikit-learn objects

All scikit-learn transformers share the same three methods, and you should be able to
explain them:

- `fit(X)` learns the parameters from the data. SimpleImputer learns the median of each
  column. StandardScaler learns the mean and std of each column.
- `transform(X)` applies what was learned.
- `fit_transform(X)` does both in one call. That is what your notebook uses.

| Object | Used for |
|---|---|
| `SimpleImputer(strategy="median")` | Fills numeric NaN with the column median. |
| `SimpleImputer(strategy="most_frequent")` | Fills categorical NaN with the column mode. |
| `StandardScaler()` | z = (x - mean) / std, so each column ends with mean 0 and std 1. |
| `OrdinalEncoder(categories=[...])` | Maps ordered levels to 0, 1, 2, ... in the order you supply. |

The `categories=[...]` argument matters: without it, OrdinalEncoder sorts alphabetically,
which would give Bachelors=0, High School=1, Masters=2, PhD=3 and destroy the real order.
You passed the order explicitly. That is a likely question.

### 2.4 The helper functions you wrote

You wrote nine helpers so the before and after EDA are identical in method and therefore
comparable:

| Helper | Returns |
|---|---|
| `numeric_cols(df)` / `categorical_cols(df)` | Column name lists, split by dtype. |
| `missing_report(df)` | Per column: missing count, missing percent, dtype, number of unique values. |
| `quality_overview(df, name)` | One column of headline numbers: rows, columns, duplicates, missing cells, memory. |
| `numeric_summary(df)` | `describe()` plus skew and kurtosis. |
| `plot_numeric_distributions` | Grid of histograms, each titled with its skew. |
| `plot_boxplots` | Grid of boxplots, for visual outlier scanning. |
| `plot_categorical_counts` | Bar chart of level frequencies, with missing values shown as their own bar. |
| `plot_correlation` | Correlation heatmap. |
| `outlier_report(df, cols)` | Tukey fences and outlier counts per column. |

If asked why you wrote helpers instead of repeating code: because the before and after
comparison is only valid if both passes are measured the same way, and because the same
functions then work on both datasets.

---

## PART 3 - The workflow, and why the order matters

This is the sequence in your notebook, for both datasets:

```
1.  Load the raw file with no cleaning arguments
2.  EDA pass 1 (structure, missing, statistics, distributions, relationships)
3.  Write down the issues, with evidence for each
4.  Fix structure    : strip whitespace, sentinels to NaN, correct the dtypes
5.  Remove duplicates
6.  Standardise inconsistent labels
7.  Apply domain rules: impossible values to NaN
8.  Apply cross-column rules: experience cannot start before age 16
9.  Handle outliers  : detect with IQR, cap at the 1st and 99th percentile
10. Impute           : median for numeric, mode for categorical
11. Encode           : ordinal, binary, one-hot, frequency
12. Scale            : StandardScaler
13. EDA pass 2 and the before/after comparison
```

**Why this order and not another one.** Expect this question, it is the easiest way for
an examiner to check whether you understand or memorised.

- **Sentinels before everything.** Until "unknown" becomes NaN, the column is text, so no
  statistic, no plot and no rule can be computed on it.
- **Duplicates before imputation.** A duplicated row votes twice. It shifts the median
  and it can change which category is the mode, so the values you impute would be
  computed from data that is counted twice.
- **Domain rules before imputation.** This is the big one. A BMI of 204 and a salary of
  -21,818 are inside the data when you compute a median. Remove them first, then compute
  the median from clean values only. If you impute first you fill the gaps with numbers
  that were partly produced by the corrupt values.
- **Imputation before encoding.** One-hot encoding a column that still has NaN either
  drops those rows or creates a NaN dummy column.
- **Encoding before scaling.** The scaler needs numbers. There is nothing to standardise
  while the column is still text.
- **Drop the id before encoding.** `id` is unique for every row. It carries no
  information, and if left in it would be scaled and fed to a model as if it meant
  something.

---

## PART 4 - Dataset 2 (the attached csv), step by step

### 4.1 What the dataset is

5050 rows, 20 columns, one row per person. Columns: `id`, `age`, `gender`, `education`,
`marital_status`, `workclass`, `occupation`, `hours_per_week`, `capital_gain`,
`capital_loss`, `native_country`, `annual_income`, `loan_status`, `experience_years`,
`job_satisfaction`, `bmi`, `smoking_status`, `physical_activity`, `credit_score`,
`disease_risk`.

`disease_risk` (0 or 1) is the natural target.

### 4.2 EDA before cleaning - what it showed

| Check | Result |
|---|---|
| `info()` | `annual_income` and `credit_score` are text, everything else numeric or text as expected |
| `describe()` | Only 9 numeric columns appear, the two text ones are absent |
| Missing values | 1509 cells over 6 columns, about 5 percent each |
| Duplicates | 49 identical rows, 50 repeated `id` values |
| Histograms | `annual_income`, `bmi`, `hours_per_week` all have long right tails |
| Boxplots | Clear isolated points far from the box on those same three columns |
| Category bars | `native_country` has 8 levels, but three of them are the same country |
| Target | `disease_risk` is 68.9 percent zeros, 31.1 percent ones |

### 4.3 The 11 issues and the fix for each

| # | Issue | Evidence | Fix |
|---|---|---|---|
| 1 | "unknown" used as a missing marker | 101 cells in `annual_income`, 101 in `credit_score`, both columns loaded as text | Replace with NaN, then `pd.to_numeric` |
| 2 | Real missing values | workclass 252, occupation 252, native_country 254, annual_income 251, bmi 250, credit_score 250 | Median for numeric, mode for categorical |
| 3 | Duplicates | 49 identical rows, 50 repeated ids | `drop_duplicates`, keep first |
| 4 | Inconsistent labels | `us` 613, `USA` 606, `U.S.` 590 are one country | Map all three to `United-States` |
| 5 | Impossible values | hours_per_week max 391, bmi range 9.2 to 204, 12 negative salaries | Set to NaN, then impute |
| 6 | Extreme skew | annual_income skew 16.09, max 2,292,645 | Cap at the 1st and 99th percentile |
| 7 | Cross-column inconsistency | 1939 rows with experience_years > age - 16 | Cap at age - 16, keep an `experience_flag` column |
| 8 | Useless identifier | `id` unique per row | Drop |
| 9 | Different scales | income in the tens of thousands, job_satisfaction 1 to 5 | StandardScaler |
| 10 | Text columns | 11 of them | Ordinal, binary and one-hot encoding |
| 11 | Class imbalance | 68.9 / 31.1 | Documented, to be handled at modelling time |

### 4.4 The cleaning, step by step, with the numbers

**Step 1 - sentinels and dtypes.** Every text column is stripped of spaces, then a list of
sentinels (`unknown`, `?`, `NA`, `n/a`, `None`, empty string, `-` and others) is replaced
with NaN, then `annual_income` and `credit_score` are converted with
`pd.to_numeric(errors="coerce")`. `errors="coerce"` means anything that still cannot be
parsed becomes NaN instead of raising an error.

Result: both columns become `float64`. Missing goes from 251 to 352 in `annual_income`
and from 250 to 351 in `credit_score`, because the 101 "unknown" cells are now counted
honestly.

**Step 2 - duplicates.** `drop_duplicates()` on the whole row, then again on `id`.
50 rows removed, 5050 becomes 5000.

Why also on `id`? Because `id` is a primary key. Two rows with the same id are the same
person even if another column differs, and one of the two has to be wrong.

**Step 3 - country labels.** `us`, `USA`, `U.S.` all map to `United-States`; `UK` maps to
`United-Kingdom`. Cardinality drops from 8 to 6.

Why it matters: one-hot encoding would otherwise create three separate columns for one
country, splitting the evidence three ways and adding two useless features.

**Step 4 - domain rules.** Values outside a sensible range become NaN:

| Column | Rule | Rows hit |
|---|---|---|
| `hours_per_week` | must be 1 to 168 | 20 |
| `bmi` | must be 12 to 60 | 26 |
| `annual_income` | must not be negative | 12 |
| `credit_score` | must be 300 to 850 | 0 |
| `age` | must be 16 to 100 | 0 |

The last two found nothing, and that is worth saying out loud: a rule that fires zero
times is still a check you performed and can prove.

Then the cross-column rule: `experience_years > age - 16` for 1926 rows (1939 before the
duplicates were dropped). Those are capped at `age - 16`, and `experience_flag` records
which rows were changed.

Why 16: it is the standard minimum working age, and it is the tightest defensible bound.
Why cap instead of delete: 1926 rows is 38 percent of the dataset, deleting them would
throw away every other column for those people. Why keep a flag: so the information that
the row was repaired is not lost, and a model can use it if it is predictive.

**Step 5 - outliers.** `outlier_report` gives the Tukey fences. After the domain rules
only two columns still have outliers: `annual_income` 1.03 percent and `bmi` 0.49 percent.

Capping limits actually used (1st and 99th percentile):

| Column | Lower | Upper |
|---|---|---|
| annual_income | 11,707 | 134,961 |
| capital_gain | 483 | 49,618 |
| capital_loss | 110 | 9,902 |

**The log transform test.** This is your strongest single talking point, because you
tested instead of assuming:

| Treatment | Skew of annual_income |
|---|---|
| Raw, invalid values removed | 16.07 |
| log1p of raw | -1.55 |
| Capped at 1/99 | 0.08 |
| log1p of capped | -1.31 |

Capping alone gives a skew of 0.08, which is as symmetric as you could ask for. Applying
log on top of that overshoots to -1.31, that is, it creates a left skew that was not there
before. So no log column was added for this dataset. (In dataset 1 the log transform is
used, because there the shape of the problem is different. Being able to explain why the
same tool is right in one place and wrong in the other is the point.)

**Step 6 - imputation.** Four numeric columns are imputed with the median
(`hours_per_week`, `annual_income`, `bmi`, `credit_score`) and three categorical columns
with the mode (`workclass`, `occupation`, `native_country`). Missing cells go to 0.

Why median and not mean: the columns are skewed and the mean is pulled by the tail.
Why mode for text: mean and median are undefined for text.

**Step 7 - encoding.** Three methods, chosen per column:

| Method | Columns | Reason |
|---|---|---|
| Ordinal | `education` (High School 0, Bachelors 1, Masters 2, PhD 3), `smoking_status` (non-smoker 0, occasional 1, smoker 2), `physical_activity` (low 0, medium 1, high 2) | The levels have a real order, one integer column preserves it |
| Binary | `loan_status` (denied 0, approved 1) | Two levels, one column is enough |
| One-hot with `drop_first=True` | `gender`, `marital_status`, `workclass`, `occupation`, `native_country` | Nominal, no order to preserve |

`job_satisfaction` is already 1 to 5 and needs nothing.

20 columns become 34.

**Step 8 - scaling.** StandardScaler on the eight continuous columns: `age`,
`hours_per_week`, `capital_gain`, `capital_loss`, `annual_income`, `experience_years`,
`bmi`, `credit_score`. Each ends with mean 0 and std 1.

The dummy columns and the ordinal codes are deliberately left alone. Scaling a 0/1 dummy
does not add information, and it makes the output harder to read.

### 4.5 EDA after cleaning - the before/after table

| Measure | Before | After |
|---|---|---|
| Rows | 5050 | 5000 |
| Columns | 20 | 34 |
| Text columns | 11 | 0 |
| Duplicate rows | 49 | 0 |
| Missing cells | 1509 | 0 |
| Memory | 3.4 MB | 1.4 MB |

Column by column:

| Feature | Mean before | Mean after | Max before | Max after | Skew before | Skew after |
|---|---|---|---|---|---|---|
| age | 43.62 | 43.59 | 69 | 69 | -0.01 | 0.00 |
| hours_per_week | 53.85 | 52.64 | 391 | 99 | 2.72 | -0.05 |
| annual_income | 74,888 | 70,615 | 2,292,645 | 134,961 | 16.09 | 0.08 |
| bmi | 27.45 | 27.08 | 204 | 47.8 | 9.04 | 0.03 |
| credit_score | 579.55 | 580.13 | 849 | 849 | -0.06 | -0.08 |
| experience_years | 21.97 | 16.42 | 44 | 44 | 0.01 | 0.50 |

Read this table out loud in the viva if you get the chance. Two things to notice:

1. `age` and `credit_score` barely move. That is the proof that the cleaning was
   targeted and did not disturb the healthy columns.
2. `experience_years` moves a lot, mean 21.97 down to 16.42, because 1926 rows were
   capped. Be ready to defend that, the defence is in 4.4 and in Part 7.

### 4.6 The finding worth volunteering

After cleaning, the strongest correlation between any feature and `disease_risk` is
about 0.04 (`marital_status_Married` at -0.039, `marital_status_Single` at +0.024).
Everything else is smaller.

In plain terms: **nothing in this dataset predicts the target.** Combined with the almost
perfectly uniform category counts (gender 2528 / 2435 / 87, education 1303 / 1260 / 1247 /
1240), this says the data was randomly generated rather than collected from real people.

This is a good thing to say, not a bad one. It shows you looked at the result instead of
stopping at "the code ran". Phrase it as: "the cleaning worked, but the data itself has
no signal, which suggests it is synthetic. On a real dataset I would expect BMI and
smoking status to show some relationship with a disease risk label."

---

## PART 5 - Dataset 1 (Census-Income KDD), step by step

### 5.1 What the dataset is

US Census Bureau data from the 1994 and 1995 Current Population Survey, published on the
UCI Machine Learning Repository as dataset 117. The training file has 199,523 rows and 42
columns: 40 attributes, one survey weight, and the label. The label is whether total
person income is above 50,000 dollars.

The file ships with no header row. The column names in your notebook were transcribed
from `census-income.names`, which is included in the same archive.

### 5.2 EDA before cleaning - what it showed

| Check | Result |
|---|---|
| Shape | 199,523 x 42, 387 MB in memory |
| `isna()` | Only 874 NaN, which looks clean and is misleading |
| `"?"` count | 415,717 cells |
| `"Not in universe"` | Present in 13 columns, up to 99 percent of a column |
| Duplicates | 3229 exact rows, 46,627 once `instance_weight` is ignored |
| Money columns | 89 to 98 percent zeros, skew up to 27.8, kurtosis up to 1090 |
| Cardinality | `prev_residence_state` 50, three birth-country columns 42 each, `household_family_stat` 38 |
| Target | 93.79 percent below 50k, 6.21 percent above |
| Age | A spike at 0, because children are in the survey. 55,992 rows are under 18 |

### 5.3 The 13 issues and the fix for each

| # | Issue | Evidence | Fix |
|---|---|---|---|
| 1 | No header, undocumented schema | 42 unnamed columns | Names taken from `census-income.names` |
| 2 | Missing values written as "?" | 415,717 cells; the 4 migration columns have 99,696 each | "?" to NaN, drop columns at 40 percent or more missing, impute the rest |
| 3 | "Not in universe" in 13 columns | vet_questionnaire 99 percent, unemployment_reason 97 percent, enrolled_edu_last_wk 93.7 percent | Kept as a normal category, see below |
| 4 | Whitespace in labels | `major_occupation`, `hispanic_origin`, `citizenship` | `str.strip()` on every text column |
| 5 | Duplicate records | 46,627 ignoring the weight; documentation says 46,716 | Drop duplicates |
| 6 | `instance_weight` is not a feature | It is a survey post-stratification weight | Dropped |
| 7 | Nominal codes stored as integers | industry_code 52 values, occupation_code 47, own_business 3, vet_benefits 3, year 2 | Cast to text, then encoded as categories |
| 8 | Zero inflation and skew | wage_per_hour 94.3 percent zeros, capital_gains 96.3, capital_losses 98.0, dividends 89.4 | log1p plus a 0/1 "has value" flag |
| 9 | Very high cardinality | 352 levels across 9 columns: states 50, birth countries 42 each, industry_code 52 | Frequency encoding |
| 10 | Different scales | age 0 to 90, dividends 0 to 99,999 | StandardScaler |
| 11 | Severe class imbalance | 93.79 / 6.21 | Documented, accuracy is the wrong metric |
| 12 | Target is text with a trailing dot | `"- 50000."`, `"50000+."` | Mapped to 0 and 1 |
| 13 | Children in the sample | 55,992 rows under 18, education level "Children" | Kept, with an `is_minor` flag |

### 5.4 The three points an examiner is most likely to probe

**Why "Not in universe" was kept as a category, not treated as missing.**

In survey design, "universe" means the set of people a question applies to. "Not in
universe" means the question was never asked of this person, because it could not apply.
`unemployment_reason` is "Not in universe" for 97 percent of rows because 97 percent of
people were not unemployed. That is not a gap in the data, it is a fact about the person.
Imputing it with the mode would invent a reason for unemployment for people who were
never unemployed. So it is kept as its own level, and the one-hot encoder gives it its
own column. In the final correlation output you can see
`class_of_worker_Not in universe` at -0.174, which means that level really does carry
information about income.

**Why `instance_weight` was dropped.**

It is a sampling weight. The census does not survey everyone, so each record is given a
weight saying how many people in the real population it represents. It is a property of
the sampling design, not of the person. The documentation that ships with the dataset
explicitly says it should not be used by a classifier and marks it "ignore". Leaving it
in would let a model learn from the survey's design rather than from the person's
attributes. It is also the reason the duplicate count changes so much: identical people
often differ only in this one column, which is why you computed duplicates after dropping
it, and got 46,627 instead of 3229.

**Why frequency encoding for the high-cardinality columns.**

Those nine columns hold 352 distinct levels between them (industry_code 52, states 50,
occupation_code 47, three birth-country columns at 42 each, household_family_stat 38,
major_industry 24, major_occupation 15). One-hot would therefore have added about 343
columns after drop_first, almost all of them nearly empty. That is the curse of
dimensionality: more columns than the data can support, each carrying very little.
Frequency encoding replaces each level with how often it occurs, so one column instead of
fifty. Target encoding (replace each level with the mean of the target for that level) is the
other option and is usually stronger, but it leaks label information into the features and
needs cross-fold fitting to be safe. Frequency encoding uses no label information at all,
so it cannot leak.

Its weakness, which you should volunteer before it is pointed out: two different states
with the same frequency get the same code and become indistinguishable.

### 5.5 The cleaning, step by step

| # | Step | Result |
|---|---|---|
| 1 | Strip whitespace, "?" to NaN, target to 0/1 | 415,717 cells become honest NaN |
| 2 | Drop `instance_weight`, drop duplicates | 46,627 rows removed, 199,523 becomes 152,896 |
| 3 | Drop the 4 migration columns (49.02 percent missing each) | 41 columns become 37 |
| 4 | Re-check duplicates after the column drop | 528 more rows removed, 152,368 remain |
| 5 | Mode-impute 5 categorical columns, add `_was_missing` flags | hispanic_origin 848, prev_residence_state 669, father_birth_country 6383, mother_birth_country 5810, birth_country 3322 |
| 6 | Cast the 5 code columns to text | They are now treated as categories |
| 7 | log1p on the 4 money columns, add `has_` flags and `is_minor` | Skew falls sharply, see the table below |
| 8 | Encode: ordinal, binary, one-hot, frequency | 47 columns become 105 |
| 9 | StandardScaler on the continuous and frequency columns | Mean 0, std 1 |

Step 4 is worth explaining because it looks odd at first: after four columns are removed,
rows that differed only in those columns become identical, so the duplicate check has to
run again.

**The log transform here.**

| Column | Skew before | Skew after | Kurtosis before | Kurtosis after |
|---|---|---|---|---|
| wage_per_hour | 8.94 | 3.29 | 155.2 | 8.9 |
| capital_gains | 18.99 | 4.33 | 393.1 | 17.1 |
| capital_losses | 7.63 | 6.03 | 61.6 | 34.4 |
| dividends | 27.79 | 2.60 | 1090.6 | 5.5 |

Why `log1p` and not `log`: these columns are almost all zeros, and log(0) is undefined.
`log1p(x)` computes log(1 + x), so 0 maps to 0 and nothing breaks.

Why the skew does not reach 0: because the problem is not only a long tail, it is that
90 to 98 percent of the values are exactly zero. No transform can fix that, which is
exactly why each column also got a `has_` flag that says whether the person had any at
all. In the final correlations, `has_dividends` at 0.251 is almost as informative as the
amount itself at 0.267.

**Encoding summary for dataset 1**

| Method | Applied to |
|---|---|
| Ordinal | `education`, 17 levels from "Children" (0) to "Doctorate degree(PhD EdD)" (16) |
| Binary | `sex` (Female 0, Male 1), and the target |
| One-hot, int8 | The 18 categorical columns with 10 levels or fewer |
| Frequency | The 9 columns with more than 10 levels |

`int8` instead of the default: a dummy column only ever holds 0 or 1, so one byte per
value is enough instead of eight. On 152,368 rows and about 90 dummy columns that is the
difference between roughly 110 MB and 14 MB.

### 5.6 What the cleaned census data shows

Unlike dataset 2, this one has real signal. Strongest correlations with earning above 50k:

| Feature | Correlation |
|---|---|
| education | +0.272 |
| dividends | +0.267 |
| has_dividends | +0.251 |
| capital_gains | +0.231 |
| weeks_worked | +0.224 |
| sex (Male = 1) | +0.178 |
| industry_code_freq | -0.190 |
| class_of_worker_Not in universe | -0.174 |
| marital_status_Never married | -0.142 |
| is_minor | -0.108 |

All of these make sense, which is the best evidence that the encoding did not scramble
anything: more education, more weeks worked and investment income go with higher income;
being a minor or not being in the labour force goes with lower income.

Note the negative frequency correlations: a large `industry_code_freq` means the person
works in a *common* industry, and common industries pay less on average. That is a real
effect, not an artefact.

---

## PART 6 - The concepts in depth

### 6.1 Missing data

**Three mechanisms** (textbook terms, worth knowing by name):

- **MCAR**, missing completely at random: the gap has nothing to do with any value. A
  sensor dropped a reading. Safe to impute, safe even to delete.
- **MAR**, missing at random: the gap depends on other observed columns. Older people skip
  the income question more often. Imputation can use the other columns.
- **MNAR**, missing not at random: the gap depends on the missing value itself. High
  earners refuse to state income. This is the dangerous one, because any imputation biases
  the result. The honest treatment is to keep a missing indicator.

In dataset 2 the missing values are spread evenly, about 5 percent in six unrelated
columns, which is consistent with MCAR. In dataset 1 the migration columns are missing
for half the sample in a structured way, which is why they were dropped rather than
imputed, and why the remaining imputed columns got `_was_missing` flags.

**Options for handling missing values**

| Method | Pros | Cons | Used here? |
|---|---|---|---|
| Drop the rows | Simple, no invented data | Loses every other column in the row; 5 percent per column over 6 columns could cost 25 percent of the data | No |
| Drop the column | Right call when a column is mostly empty | Loses a whole variable | Yes, the 4 migration columns at 49 percent |
| Mean | Preserves the mean | Distorted by outliers and skew | No |
| Median | Robust to outliers and skew | Shrinks the variance | Yes, numeric |
| Mode | Only option for text | Over-feeds the majority category | Yes, categorical |
| Constant / "Unknown" as its own level | Keeps the missingness visible | Adds a level that may be meaningless | Effectively yes, for "Not in universe" |
| KNN or iterative imputation | Uses the other columns, usually more accurate | Slow on 150k rows, needs scaled inputs, easy to leak | No |

If asked "why not KNN imputation": it needs the data already scaled and encoded, it is
expensive on a 150,000 row table, and with only 5 percent missing at random the extra
accuracy would not change any conclusion in this assignment.

**Why imputation shrinks variance.** If you put the median into 250 cells, those 250 rows
now all sit exactly at the centre. The spread of the column goes down and the peak at the
median goes up. This is a genuine cost, and the honest answer is that it is accepted in
exchange for keeping the rows. The alternative, dropping rows, costs more.

### 6.2 Outliers

**Detection**

- Tukey / IQR: outside Q1 - 1.5 IQR or Q3 + 1.5 IQR. Distribution free, robust. Used here.
- Z-score: |z| > 3. Assumes roughly normal data and uses the mean and std, which the
  outliers themselves distort. Not used, and you can say why.
- Visual: boxplots and histograms, which is why the notebook plots both before and after.

**Treatment**

- Remove the row: only when the value is certainly an error and the row cannot be saved.
- Set to NaN and impute: what you did for the impossible values, because the rest of the
  row is still good.
- Cap / winsorise: pull extreme values back to a percentile. Keeps the row, keeps the
  ordering, removes the leverage. What you did for the money columns.
- Transform: log or square root, which compresses the tail rather than cutting it.
- Leave alone and use a robust model: trees do not care about the magnitude of an outlier.

**Winsorising at 1 and 99 percent** means: everything below the 1st percentile becomes the
1st percentile value, everything above the 99th becomes the 99th. You keep 100 percent of
the rows and lose only the exact magnitude of the most extreme 2 percent.

Be ready for: "isn't the 1/99 threshold arbitrary?" Yes, it is a convention. The honest
answer is that any threshold is a judgement call, that you checked its effect rather than
assuming it (skew 16.09 to 0.08), and that the alternative of capping at the Tukey fences
would have been equally defensible and would have capped slightly more values.

### 6.3 Skewness and transforms

Why skew matters:
- The mean stops describing the typical case. Mean income 74,888 while the median is
  69,995 and the max is 2.29 million.
- Standardisation is computed from the mean and std, so a skewed column scales badly.
- Linear models assume roughly symmetric errors; distance based models let the big values
  dominate.

Transforms, from mildest to strongest: square root, log, reciprocal. `log1p` is the log
variant that tolerates zeros. Box-Cox and Yeo-Johnson pick the exponent automatically;
Yeo-Johnson also handles zero and negative values.

The rule you demonstrated: **apply a transform because you measured that it helps, not
because the column is called "income"**. It helped in dataset 1 (skew 27.79 to 2.60) and
it hurt in dataset 2 (0.08 to -1.31).

### 6.4 Encoding

| Method | How it works | When to use | Risk |
|---|---|---|---|
| Label encoding | Each level gets an arbitrary integer | Only for the target, or for tree models | Invents an order that does not exist |
| Ordinal encoding | Integers in a meaningful order you supply | Ordered categories | Assumes equal gaps between levels |
| One-hot | One 0/1 column per level | Nominal with few levels | Adds many columns; dummy trap if you keep all levels |
| Frequency | Each level replaced by how often it occurs | High cardinality | Two equally common levels collide |
| Target / mean | Each level replaced by the mean target for that level | High cardinality, strong predictors | Leaks the label, needs cross-fold fitting |
| Binary / hashing | Compressed representations | Very high cardinality | Hard to interpret |

**The dummy variable trap.** If a column has 4 levels and you create 4 dummy columns, they
always add up to 1, so any one of them is perfectly predictable from the other three. That
is perfect multicollinearity. A linear model cannot then find unique coefficients, and the
matrix inversion behind it becomes singular. `drop_first=True` deletes one level, which
becomes the baseline, and every remaining coefficient is read as "compared to the baseline".

When is it safe to keep all levels? Tree models and regularised models do not break. It
matters for plain linear and logistic regression.

**Why ordinal for education but one-hot for occupation.** Education has a true rank, a PhD
is more education than a Masters, so one integer column keeps that relationship in a form
the model can use and costs one column instead of four. Occupation has no rank: Sales is
not more or less than Tech-support, so the only honest representation is a separate
indicator for each.

The cost of ordinal encoding, which you should mention before the examiner does: it
assumes the gaps are equal, that is, PhD minus Masters equals Bachelors minus High School.
That is not strictly true. It is accepted because the alternative, one-hot, throws away the
order entirely.

### 6.5 Scaling

| Method | Formula | Output | Best for |
|---|---|---|---|
| Standardisation (z-score) | (x - mean) / std | Mean 0, std 1, unbounded | Most cases, especially with outliers already handled |
| Min-max normalisation | (x - min) / (max - min) | 0 to 1 | Neural nets, image data; very sensitive to outliers |
| Robust scaling | (x - median) / IQR | Centred on the median | When outliers remain in the data |

You used standardisation, and the reason is that outliers had already been capped, so the
mean and std are now trustworthy, and standardisation does not force an artificial 0 to 1
range on columns like income.

**Which models need scaling?**

- Need it: KNN, SVM, K-means, PCA, neural networks, and any model with L1/L2
  regularisation (an unscaled column gets penalised unfairly).
- Do not need it: decision trees, random forests, gradient boosting. They split on
  thresholds, and a split point does not care about units.

Scaling is not harmful for the second group, so it is normal to scale anyway in a general
pipeline, which is what you did.

### 6.6 Class imbalance

Census: 93.79 percent versus 6.21 percent.

**The accuracy paradox.** A model that predicts "below 50k" for everyone scores 93.79
percent accuracy and is completely useless, because it never finds a single high earner.
So accuracy is the wrong metric for this dataset.

Better metrics:
- Precision: of the people you flagged as high earners, how many were.
- Recall: of the actual high earners, how many you found.
- F1: the harmonic mean of the two.
- ROC-AUC, and on heavy imbalance PR-AUC, which is more informative.

Fixes at modelling time: stratified train/test split so both sets keep the same ratio,
`class_weight="balanced"`, oversampling the minority (SMOTE), or undersampling the
majority. You did not do any of these because the assignment stops before modelling, and
resampling before the split would leak. Saying that is a strong answer.

### 6.7 The dimensions of data quality

If you are asked "how do you judge whether data is good", these six words cover it, and
you can attach an example from your own work to each:

| Dimension | Failure you found |
|---|---|
| Completeness | 1509 missing cells, 415,717 "?" cells |
| Consistency | `us` / `USA` / `U.S.`, experience longer than age minus 16 |
| Validity | BMI 204, 391 hours per week, negative income |
| Accuracy | Values that are possible but wrong; hardest to detect without a reference |
| Uniqueness | 49 and 46,627 duplicates |
| Timeliness | The census data is from 1994-95, so it is not current |

---

## PART 7 - Weak points, and the honest answer for each

Every one of these is a real criticism. Do not pretend otherwise. The examiner is testing
whether you know the limits of your own work, and admitting a limit with a reason is worth
more than defending everything.

**1. You fitted the imputer and the scaler on the whole dataset. That is data leakage.**

Correct, and this is the most likely technical challenge. The honest answer:

> Strictly, yes. If I were training a model, the median, the mode, the scaler's mean and
> std and the frequency encoding map all have to be learned from the training split only
> and then applied to the test split, otherwise information from the test set reaches the
> model. Here there is no train/test split, because the assignment is a cleaning and EDA
> task, and the cleaned table is the deliverable. If this fed a model I would put these
> steps inside a scikit-learn Pipeline with a ColumnTransformer and fit it inside the
> cross-validation loop.

That answer converts a weakness into evidence that you understand the wider workflow.

**2. Capping experience changed 1926 rows, a third of the dataset.**

> The original values were impossible, not merely extreme, so leaving them would encode a
> relationship that cannot exist. Capping at age minus 16 keeps the ordering and is the
> tightest defensible bound. I kept `experience_flag` so no information about which rows
> were repaired is lost, and so a model can test whether those rows behave differently. The
> alternative was to null the values and impute them, which would have replaced a known
> upper bound with a guess.

**3. Dropping 46,627 census rows as duplicates may be wrong.**

> It is a judgement call. With 40 attributes, two people can genuinely share every value,
> particularly when many columns are "Not in universe" for children. On the other side, the
> dataset documentation itself reports 46,716 "duplicate or conflicting instances", so the
> authors regard them as a known defect, and duplicated rows would give those profiles extra
> weight in any statistic or model. I dropped them and recorded the count so the decision is
> visible and reversible.

**4. Mode imputation biases the majority category.**

> Yes. Filling 6383 missing birth countries with the most common one makes that country more
> dominant than it really is. That is why I added a `_was_missing` flag for every imputed
> column in the census data, so the model can separate real values from filled ones. For
> dataset 2 the missing share was about 5 percent and looked random, so I accepted the
> simpler treatment.

**5. Frequency encoding loses identity.**

> Two states with the same frequency get the same number and become indistinguishable. I
> accepted that because one-hot encoding those nine columns would have added about 343
> nearly empty columns. Target encoding would preserve more, but it leaks label information
> unless it is fitted inside cross-validation.

**6. The 40 percent missing threshold is arbitrary.**

> It is a rule of thumb. The four columns it removed were all at 49 percent, so any
> threshold between about 20 and 45 percent gives the same result. Nothing else in the
> dataset was near the line.

**7. Dataset 2 looks synthetic, so the cleaning does not prove anything about real data.**

> Agreed, and I said so in the report. The value of the exercise is the process: the same
> rules applied to a real dataset would find real problems. The evidence for it being
> synthetic is that no feature correlates with the target above about 0.04 and the
> categories are almost perfectly uniform.

**8. Why not use a scikit-learn Pipeline for the whole thing?**

> A Pipeline is the right structure for production and for cross-validation, because it
> guarantees the transforms are fitted on the training fold only. I wrote the steps out
> explicitly because the assignment asks me to show and explain each cleaning decision, and
> a Pipeline hides them. The step log in the notebook is the record of what was applied and
> in what order.

---

## PART 8 - Rapid fire questions and answers

Cover the answers and work down the list. If you can answer 40 of these you are ready.

### On the data

**Q1. What are your two datasets and how big are they?**
Census-Income KDD from UCI, 199,523 rows by 42 columns. The attached csv, 5050 rows by 20
columns.

**Q2. What is the target in each?**
`income_label` (above or below 50k) in the census data, `disease_risk` (0/1) in the
attached data.

**Q3. How did you know `annual_income` had a problem before you read any value?**
`df.info()` showed it as a text column, and it was missing from `df.describe()`. A column
of salaries can only be text if some cell is not a number.

**Q4. How many missing values were in each dataset?**
Dataset 2: 1509 NaN plus 202 "unknown" strings, 1711 in total. Dataset 1: 874 NaN plus
415,717 "?" cells.

**Q5. Why does pandas report only 874 missing values for the census file?**
Because the real missing values are written as "?", which pandas treats as an ordinary
string. The 874 come from `hispanic_origin`, which has a legitimate level spelled "NA",
and "NA" is in the default `na_values` list, so pandas converts it to NaN by mistake.

**Q6. What is a sentinel value?**
A normal looking value used to mean "missing": "unknown", "?", "NA", -999, 0. Dangerous
because software treats it as real data.

### On cleaning decisions

**Q7. Why median and not mean for imputation?**
The columns are skewed. The mean is pulled toward the tail; the median is not.

**Q8. Why mode for the text columns?**
Mean and median are undefined for text. The mode is the only summary of a categorical
column.

**Q9. Why did you not just delete the rows with missing values?**
Six columns each have about 5 percent missing, but on different rows. Dropping every row
with at least one missing value would cost 1488 rows, 29.5 percent of the dataset, and
throw away all the good columns in those rows.

**Q10. Why remove duplicates before imputing?**
A duplicated row is counted twice, so it shifts the median and can change which value is
the mode. The imputed values would then come from data counted twice.

**Q11. Why set impossible values to NaN before imputing rather than after?**
Because the median is computed from the column. If a BMI of 204 and a salary of -21,818
are still present, they contaminate the very number you are about to insert.

**Q12. Why cap outliers instead of deleting them?**
Deleting loses the whole row including its good columns. Capping keeps every row, keeps
the ordering, and only removes the extreme magnitude.

**Q13. Why 1st and 99th percentile?**
It is a convention that removes the leverage of the most extreme 2 percent. I checked the
effect: skew fell from 16.09 to 0.08. Tukey fences would have been an equally defensible
choice.

**Q14. Why did you not log transform income in dataset 2?**
I tested it. Capping alone gave skew 0.08. Adding the log pushed it to -1.31, which
creates a left skew that was not there. So the log was dropped for that dataset.

**Q15. Then why use log1p in the census data?**
Different problem. Those columns are 89 to 98 percent zeros with skew up to 27.8, so the
tail is genuinely long rather than a handful of corrupt values, and log1p brought the
worst one from 27.79 to 2.60.

**Q16. Why log1p and not log?**
The columns contain zeros and log(0) is undefined. log1p computes log(1 + x), so 0 maps
to 0.

**Q17. Why did you add the `has_capital_gains` style flags?**
Because no transform can fix a column that is 96 percent zeros. The flag separates "had
any at all" from "how much", and it turned out to be almost as predictive: `has_dividends`
correlates 0.251 with the target and the amount correlates 0.267.

**Q18. What is `errors="coerce"`?**
An argument to `pd.to_numeric`. Anything that cannot be converted becomes NaN instead of
raising an exception.

**Q19. What does `drop_duplicates(keep="first")` keep?**
The first occurrence of each duplicated row, and removes the later copies.

**Q20. Why did you also drop duplicates on `id` alone?**
`id` is the primary key. Two rows with the same id are the same person, so at least one of
them is wrong even if the other columns differ.

### On encoding

**Q21. Name the encoding methods you used and where.**
Ordinal for education, smoking status and physical activity; binary for loan status and
sex; one-hot with drop_first for the nominal columns; frequency encoding for the nine
high-cardinality census columns.

**Q22. Difference between label and one-hot encoding?**
Label encoding puts one arbitrary integer per level in a single column, which invents an
order. One-hot creates a separate 0/1 column per level and invents nothing.

**Q23. What is the dummy variable trap?**
If you keep one dummy per level they sum to 1, so any one is perfectly predictable from
the others. That is perfect multicollinearity and a linear model cannot fit unique
coefficients. `drop_first=True` avoids it.

**Q24. Does it matter for a random forest?**
No. Trees split on thresholds and are not affected by collinearity. It matters for linear
and logistic regression.

**Q25. Why ordinal for education and not one-hot?**
Education has a real order, so one integer column keeps the ranking and costs one column
instead of four. Occupation has no order, so it gets one-hot.

**Q26. What does ordinal encoding assume that is not strictly true?**
Equal gaps between consecutive levels, that PhD minus Masters equals Masters minus
Bachelors.

**Q27. Why did you pass `categories=` to OrdinalEncoder?**
Without it, scikit-learn orders the levels alphabetically, which would give Bachelors=0
and High School=1, destroying the real order.

**Q28. What is frequency encoding and why use it?**
Each level is replaced by how often it appears. One column instead of fifty for states.
Used because the nine high-cardinality columns hold 352 levels between them, so one-hot
would have added about 343 nearly empty columns.

**Q29. What is the risk of target encoding?**
It uses the label to build the feature, so it leaks target information unless it is fitted
inside cross-validation folds.

**Q30. Why int8 for the dummies?**
A dummy is only 0 or 1, so one byte is enough instead of eight. On 152,368 rows it is the
difference between about 110 MB and 14 MB.

### On scaling and statistics

**Q31. What does StandardScaler do?**
z = (x - mean) / std, per column. The result has mean 0 and standard deviation 1.

**Q32. Standardisation versus normalisation?**
Standardisation centres on the mean with unit variance and is unbounded. Min-max
normalisation squeezes into 0 to 1 and is very sensitive to outliers.

**Q33. Which models need scaling?**
KNN, SVM, K-means, PCA, neural networks, and regularised linear models. Trees and
boosting do not.

**Q34. Why scale after encoding, not before?**
There is nothing to scale while the column is text. Encoding produces the numbers that
scaling then standardises.

**Q35. What is skewness, and what counts as high?**
A measure of asymmetry. 0 is symmetric, positive means a long right tail. Above 1 in
absolute value is substantial. Your income column was 16.09.

**Q36. What is kurtosis?**
Tail heaviness. Pandas reports excess kurtosis, so a normal distribution gives 0. Census
dividends was 1090 before the transform.

**Q37. What does IQR mean and why is it robust?**
Q3 minus Q1, the width of the middle 50 percent. It is built from quartiles rather than
the mean and std, so extreme values cannot inflate it.

**Q38. What does a boxplot show?**
The median, the box from Q1 to Q3, whiskers out to 1.5 IQR, and individual points beyond
that as outliers.

**Q39. What does a correlation of 0.27 mean?**
A weak to moderate positive linear relationship. Education is the strongest single
predictor of high income in the census data, but on its own it explains only a small part
of it.

**Q40. Can correlation be 0 when a relationship exists?**
Yes. Pearson correlation only detects linear relationships. A U shaped relationship can
give 0.

### On the results

**Q41. What did the census data turn out to say?**
More education, more weeks worked and investment income go with earning above 50k;
being a minor, never married or not in the labour force go with earning below it.
All the correlations point in directions that make sense.

**Q42. What did the attached dataset turn out to say?**
Nothing. No feature correlates with `disease_risk` above about 0.04, and the categories
are almost perfectly uniform, which suggests the data was randomly generated.

**Q43. Why is 93.79 percent accuracy a bad result on the census data?**
Because predicting "everyone earns under 50k" scores exactly that and finds no high
earners at all. With that imbalance you need precision, recall, F1 or PR-AUC.

**Q44. How would you handle that imbalance?**
Stratified splitting, class weights, or resampling such as SMOTE, and all of it applied
after the train/test split so nothing leaks.

**Q45. How do you prove the cleaning worked?**
The before and after tables: missing cells 1509 to 0, duplicates 49 to 0, text columns 11
to 0, income skew 16.09 to 0.08, BMI max 204 to 47.8, hours max 391 to 99. And the columns
that were already healthy barely moved, age mean 43.62 to 43.59.

**Q46. How do you know you did not destroy the signal while cleaning?**
The shape of the sensible part of each distribution is unchanged in the before and after
histograms, only the impossible tail is gone, and the untouched columns kept their
statistics.

**Q47. Why did the census row count fall from 199,523 to 152,368?**
46,627 duplicates removed, then 528 more that only became identical after the four
migration columns were dropped.

**Q48. Why did the column count rise from 47 to 105?**
One-hot encoding of the 18 low-cardinality categorical columns, which turns one column
into several 0/1 columns.

**Q49. Why is the cleaned census file smaller in memory despite having more columns?**
Text columns are expensive. Replacing 29 text columns with int8 dummies and float32
numbers took it from 387 MB to 38 MB.

**Q50. If you had more time, what would you do next?**
Put every transform inside a scikit-learn Pipeline fitted on a training split only, then
train a baseline logistic regression and a gradient boosting model on the census data,
compare them on PR-AUC rather than accuracy, and test whether the missing-value flags
carry any predictive value.

---

## PART 9 - The numbers to memorise

If you only memorise one page, memorise this one.

**Dataset 2, the attached csv**

| | |
|---|---|
| Raw shape | 5050 x 20 |
| Final shape | 5000 x 34 |
| "unknown" cells | 101 + 101 |
| Missing cells | 1509 (1711 including the sentinels) |
| Duplicates | 49 identical rows, 50 repeated ids |
| Country labels | 8 down to 6 |
| Impossible values | 20 hours > 168, 26 BMI outside 12-60, 12 negative salaries |
| Experience capped | 1926 rows |
| Income skew | 16.09 before, 0.08 after |
| BMI max | 204 before, 47.8 after |
| Hours max | 391 before, 99 after |
| Target | 68.9 / 31.1 |
| Strongest correlation with target | about 0.04, effectively none |

**Dataset 1, Census-Income KDD**

| | |
|---|---|
| Raw shape | 199,523 x 42 |
| Final shape | 152,368 x 105 |
| "?" cells | 415,717 |
| Duplicates | 3229 exact, 46,627 ignoring the weight |
| Columns dropped | 4 migration columns at 49 percent missing, plus instance_weight |
| Imputed | 5 categorical columns, each with a `_was_missing` flag |
| Money columns zeros | 89.4 to 98.0 percent |
| Worst skew | dividends 27.79 before, 2.60 after |
| Encoding split | 18 one-hot, 9 frequency, 1 ordinal (17 levels), 1 binary |
| Target | 93.79 / 6.21 |
| Strongest correlation | education +0.272, dividends +0.267 |
| Memory | 387 MB before, 38 MB after |

---

## PART 10 - How to behave in the viva

**Say the number.** "About 5 percent missing" is fine, "252 in workclass" is better. You
have the numbers, use them, they are the difference between having done the work and
having watched it happen.

**Give the reason, not just the action.** Never say only "I used the median". Say "I used
the median because the column is skewed and the mean would be dragged by the tail".

**Volunteer a limitation once.** Saying "the imputer was fitted on the full dataset, which
would be leakage if there were a train/test split" before you are asked is the strongest
move available. It shows you know where the boundary is.

**If you do not know, say so and reason out loud.** "I have not used that, but it sounds
like it would do X, and the trade-off would be Y" beats silence and beats a guess
delivered confidently.

**Things not to say**
- "I dropped the outliers" - you did not, you capped them, and the difference matters.
- "I normalised the data" - you standardised it. Normalisation usually means min-max.
- "Not in universe means missing" - it means the question did not apply.
- "Accuracy would be 94 percent so the model is good" - that is the accuracy paradox.
- "I removed the missing values" - be precise: you imputed most and dropped four columns.

**If a question goes somewhere you did not cover**, bring it back to what you can defend:
"I did not test that in this assignment, but the check I did run was X, and it showed Y."

---

## Where each answer lives in the notebook

| Section in the notebook | What it covers |
|---|---|
| Cell 1-2 | Header, imports, helper functions |
| A.1 to A.8 | Dataset 2, EDA before cleaning |
| A.9 | The 11 issues, with printed evidence |
| A.10, steps 1-8 | Dataset 2 cleaning |
| A.11 | Dataset 2, EDA after, plus the before/after tables and plots |
| B.1 to B.9 | Dataset 1, EDA before cleaning |
| B.10 | The 13 issues, with printed evidence |
| B.11, steps 1-9 | Dataset 1 cleaning |
| B.12 | Dataset 1, EDA after, plus the comparison |
| Part C | Both datasets side by side, cleaned files written, conclusion |


---

## One correction to know before you go in

The notebook comment and the report both say one-hot on the high-cardinality census
columns "would have added around 220 almost empty columns". The exact figure is higher:
those nine columns hold 352 levels between them, so one-hot would add about 343 columns
after `drop_first`.

The 220 figure counts only the identity-style columns (roughly 50 states plus three
birth-country columns at 42 each plus 38 household categories = 214) and leaves out
`industry_code`, `occupation_code`, `major_industry` and `major_occupation`.

If it comes up, say: "the figure in my comment counts the state and birth-country columns;
across all nine frequency-encoded columns it is about 350." The argument does not change,
it gets stronger. Do not quote 220 as the total.

---

## PART 11 - Outliers and anomalies, the complete list

"Outlier" and "anomaly" are not the same word. Keep the difference straight, it is an easy
question to ask:

- **Outlier**: a value far from the rest of the distribution. It may be perfectly correct.
- **Anomaly**: something that should not be there at all. Impossible values, broken
  relationships between columns, duplicates, disguised missing values, top-coded ceilings.

Every outlier is judged against one column. Most anomalies are only visible when you
compare two columns, or when you know something about how the data was collected.

### 11.1 Dataset 2 - outliers (Tukey/IQR rule on the raw data)

| Column | Min | Max | Lower fence | Upper fence | Below | Above | % flagged | Skew |
|---|---|---|---|---|---|---|---|---|
| age | 18 | 69 | -8.0 | 96.0 | 0 | 0 | 0.00 | -0.01 |
| hours_per_week | 5 | 391 | -43.0 | 149.0 | 0 | 20 | 0.40 | 2.72 |
| capital_gain | 3 | 49,994 | -23,392 | 74,433 | 0 | 0 | 0.00 | -0.04 |
| capital_loss | 0 | 9,998 | -5,240 | 15,083 | 0 | 0 | 0.00 | 0.03 |
| annual_income | -21,818 | 2,292,645 | 1,684 | 139,643 | 19 | 39 | 1.23 | 16.09 |
| experience_years | 0 | 44 | -22.0 | 66.0 | 0 | 0 | 0.00 | 0.01 |
| job_satisfaction | 1 | 5 | -1.0 | 7.0 | 0 | 0 | 0.00 | -0.01 |
| bmi | 9.2 | 204 | 13.5 | 40.7 | 15 | 34 | 1.02 | 9.04 |
| credit_score | 300 | 849 | 27.8 | 1,130 | 0 | 0 | 0.00 | -0.06 |

Only three columns have outliers at all: `annual_income` (58 rows), `bmi` (49 rows) and
`hours_per_week` (20 rows). The rest are completely clean by the IQR rule.

Note: the table in your notebook shows smaller percentages (income 1.03, bmi 0.49) because
it runs *after* the impossible values have already been set to NaN. Both are right, they
are measured at different points. If asked, say "1.23 percent before the domain rules,
1.03 percent after".

The extremes: top five incomes are 2,292,645 / 1,671,330 / 1,665,270 / 1,627,890 /
1,575,405, against a median of 69,995. Top five BMI are 204 / 182.5 / 158 / 143 / 137,
lowest five are 9.2 / 9.4 / 9.8 / 10.0 / 10.6.

### 11.2 Dataset 2 - anomalies

| # | Anomaly | Count | Why it is an anomaly | Handled in the notebook |
|---|---|---|---|---|
| 1 | `hours_per_week` above 168 | 20 | A week contains 168 hours. Values run 306 to 391 | Yes, set to NaN and imputed |
| 2 | BMI outside 12 to 60 | 26 (19 high, 7 low) | 204 and 9.2 are not survivable | Yes |
| 3 | Negative `annual_income` | 12 | Income cannot be below zero, min is -21,818 | Yes |
| 4 | `experience_years` > `age - 16` | 1939 | Work cannot start before 16 | Yes, capped, flagged |
| 5 | `experience_years` > `age` | 736 | Stronger version of the same thing: more years of work than years alive | Yes, covered by rule 4 |
| 6 | "unknown" as a missing marker | 202 | Turned two numeric columns into text | Yes |
| 7 | Duplicate rows / duplicate ids | 49 / 50 | Same person recorded twice | Yes |
| 8 | Three spellings of one country | 1809 rows | `us`, `USA`, `U.S.` | Yes |
| 9 | `capital_gain` > `annual_income` | 286 | Capital gains larger than total annual income. Possible in real life but unlikely 286 times | **No** |
| 10 | `capital_loss` > `annual_income` | 29 | Same idea | **No** |
| 11 | Distributions are flat | whole dataset | See below | Noted in the conclusion |

**The flatness anomaly, and why it is the most interesting one.** Look at the raw
statistics:

| Column | Min | Max | Mean | Median | Skew |
|---|---|---|---|---|---|
| age | 18 | 69 | 43.6 | 43.5 | -0.01 |
| capital_gain | 3 | 49,994 | 25,520 | 25,933 | -0.04 |
| capital_loss | 0 | 9,998 | 4,936 | 4,926 | 0.03 |
| credit_score | 300 | 849 | 579.5 | 589 | -0.06 |
| experience_years | 0 | 44 | 22.0 | 22.0 | 0.01 |
| job_satisfaction | 1 | 5 | 3.0 | 3.0 | -0.01 |

In every case the mean sits on the median, both sit at the midpoint of the range, and the
skew is effectively zero. That is a **uniform distribution**, and real human data is almost
never uniform. Real ages cluster, real credit scores are lumpy, real capital gains are zero
for most people. Combined with the fact that no feature correlates with `disease_risk`
above about 0.04, the conclusion is that this dataset was randomly generated with corrupt
values injected on purpose, which is exactly what you would do to build a cleaning
exercise.

Saying this out loud is a strong move. It shows you looked at the shape of the data and
not only at the error list.

### 11.3 Dataset 1 - outliers (Tukey/IQR rule on the raw data)

| Column | Min | Max | Lower fence | Upper fence | Flagged | % | Skew | % zeros |
|---|---|---|---|---|---|---|---|---|
| age | 0 | 90 | -37.5 | 102.5 | 0 | 0.00 | 0.4 | 1.4 |
| wage_per_hour | 0 | 9,999 | 0.0 | 0.0 | 11,304 | 5.67 | 8.9 | 94.3 |
| capital_gains | 0 | 99,999 | 0.0 | 0.0 | 7,379 | 3.70 | 19.0 | 96.3 |
| capital_losses | 0 | 4,608 | 0.0 | 0.0 | 3,906 | 1.96 | 7.6 | 98.0 |
| dividends | 0 | 99,999 | 0.0 | 0.0 | 21,141 | 10.60 | 27.8 | 89.4 |
| num_persons_worked | 0 | 6 | -6.0 | 10.0 | 0 | 0.00 | 0.8 | 48.1 |
| weeks_worked | 0 | 52 | -78.0 | 130.0 | 0 | 0.00 | 0.2 | 48.1 |
| instance_weight | 37.9 | 18,656 | -628.9 | 3,879 | 6,238 | 3.13 | 1.4 | 0.0 |

**The most important row in this table is the fence column.** For all four money columns
the lower and upper fences are both exactly 0. That happens because more than 75 percent of
the values are zero, so Q1 = Q3 = 0 and IQR = 0. The fences collapse onto zero, and the
rule then flags *every non-zero value* as an outlier.

This is a genuine failure of the IQR rule on zero-inflated data, and it is worth saying:
"the IQR rule is meaningless on those four columns because the quartiles are all zero, so I
treated them with a log transform and a has-value flag instead of an outlier rule." That
answer is worth more than any number in the table.

### 11.4 Dataset 1 - anomalies

| # | Anomaly | Count | Why it is an anomaly | Handled in the notebook |
|---|---|---|---|---|
| 1 | Missing values written as "?" | 415,717 cells | Not recognised as missing by any tool | Yes |
| 2 | Valid level "NA" auto-converted to NaN | 874 | pandas destroys a real `hispanic_origin` category by default | Yes, identified |
| 3 | "Not in universe" in 13 columns | up to 99% of a column | Structural, not missing | Yes, kept as a level |
| 4 | Duplicate records | 3,229 exact, 46,627 ignoring the weight | Same person counted twice | Yes |
| 5 | `instance_weight` is not a person attribute | all rows | Survey design leaking into the features | Yes, dropped |
| 6 | Trailing spaces in labels | 3 columns | `"Handlers equip cleaners etc "` | Yes |
| 7 | Nominal codes stored as integers | 5 columns | Model would read code 44 as bigger than code 2 | Yes |
| 8 | Children in an income survey | 55,992 under 18, 2,839 at age 0 | A different population mixed in | Yes, `is_minor` flag |
| 9 | Zero inflation | 89 to 98 percent zeros | Breaks the IQR rule, see above | Yes, log1p + has-value flags |
| 10 | Top-coded ceiling values | 390 rows at exactly 99,999 | See below | **No** |
| 11 | 0 weeks worked but wage per hour above 0 | 283 | Contradiction between two columns | **No** |
| 12 | Minors with wages or capital gains | 397 with wages, 45 with gains, 2 above 50k | Possible but unusual | Partly, via `is_minor` |
| 13 | Severe class imbalance | 93.79 / 6.21 | Accuracy becomes meaningless | Yes, documented |

**Top-coding, the anomaly you should know about even though you did not handle it.**

`capital_gains` has 390 rows at exactly 99,999. The next highest value in the whole column
is 41,310. There is nothing between 41,310 and 99,999, and then 390 people sit on the
ceiling exactly. That is not a real amount of money. It is **top-coding**: the Census
Bureau caps any value above a threshold at a fixed ceiling to protect the identity of rich
respondents. The same thing happens with `dividends` (25 rows at 99,999, next value 95,095)
and `wage_per_hour` (1 row at 9,999, next value 9,916).

So 99,999 is functionally a sentinel value, the same class of problem as "unknown" and "?",
just harder to spot because it looks like a number.

If an examiner raises it, the honest answer is:

> I did not catch that one. 99,999 in `capital_gains` is a top-coded ceiling, not a real
> amount, and 390 rows sit on it while the next real value is 41,310. It is the same kind
> of disguised sentinel as "?". The log transform I applied reduces its leverage but does
> not fix it. The proper treatment would be either a "top-coded" flag column, or treating
> those rows as censored data.

Volunteering the fix is what turns a miss into a good answer.

**The other two gaps to know.** `capital_gain` exceeds `annual_income` for 286 rows in
dataset 2, and 283 census rows report zero weeks worked with a non-zero hourly wage. Both
are cross-column contradictions of the same type as the experience-versus-age rule you did
catch. If asked whether your rule list was complete, say no, and name these three: "I
checked experience against age, but a fuller pass would also check capital gains against
income, wage against weeks worked, and the top-coded ceilings."

### 11.5 One structural fact worth knowing

In the census data, `capital_gains > 0` and `capital_losses > 0` never happen in the same
row, zero cases out of 199,523. A person is recorded as having one or the other, never
both. That is a design rule of the survey, not a coincidence, and it explains why the two
columns behave so similarly in the correlation output.

### 11.6 Summary answer if you are simply asked "what were the outliers"

> In the attached dataset, three columns had outliers by the IQR rule: `annual_income`
> with 58 rows, top value 2.29 million against a median of 70,000; `bmi` with 49 rows,
> from 9.2 up to 204; and `hours_per_week` with 20 rows between 306 and 391. Some of those
> were genuine extremes and some were impossible, so I separated them: impossible values
> were set to NaN and imputed, genuine extremes were capped at the 1st and 99th percentile.
>
> In the census data, the IQR rule does not work on the four money columns, because more
> than 75 percent of their values are zero, so both fences sit at zero and every non-zero
> value gets flagged. Instead I used log1p plus a has-value flag. The other outliers there
> are the children, 55,992 rows under 18 in an income survey, which I kept but flagged.
