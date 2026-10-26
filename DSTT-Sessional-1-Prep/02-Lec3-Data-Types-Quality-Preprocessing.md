# Part 2 — Lecture 3: Data & Attribute Types, Data Quality, Preprocessing
### ★ This is the single highest-yield lecture for Sessional-I ★

---

## 2.1 Qualitative vs Quantitative

| | Consists of |
|---|---|
| **Qualitative data** | attributes, labels, or **nonnumerical** entries |
| **Quantitative data** | **numerical** measurements or counts |

Slide example — a table of Student | GPA: **Student (Sally, Bob, Cindy, Mark, Kathy) = qualitative;
GPA (3.22, 3.98, 2.75, 2.24, 3.84) = quantitative.**

> Trap: a number is not automatically quantitative. **ID numbers and zip codes are numeric but
> qualitative/nominal** — you can't meaningfully average them.

---

## 2.2 LEVELS OF MEASUREMENT — "NOIR" (guaranteed exam material)

> **The level of measurement determines which statistical calculations are meaningful.**

| Level | Order? | Equal intervals? | True zero? | Valid operations | Examples |
|---|---|---|---|---|---|
| **Nominal** | ✗ | ✗ | ✗ | =, ≠, counts, **mode** | hair colour, marital status, occupation, ID numbers, zip codes, gender |
| **Ordinal** | ✓ | ✗ | ✗ | + **median**, percentiles, rank | Size {small, medium, large}, grades, army rankings, education level, Likert scales |
| **Interval** | ✓ | ✓ | ✗ (**no true zero**) | + **mean**, standard deviation, +/− | Temperature in **°C or °F**, calendar dates, IQ |
| **Ratio** | ✓ | ✓ | ✓ (**inherent zero**) | + ×, ÷, **ratios** | Temperature in **Kelvin**, length, counts, monetary quantities, age, weight, income |

**The killer distinction, in the slides' own words:**
- Interval: *"Measured on a scale of equal-sized units, values have order, **no true zero-point**."*
- Ratio: *"Inherent zero-point. We can speak of values as being an order of magnitude larger than the
  unit of measurement (**10 K° is twice as high as 5 K°**)."*

So: 20 °C is **not** "twice as hot" as 10 °C (interval, arbitrary zero), but 20 K **is** twice 10 K (ratio).

Mnemonic: **N-O-I-R** (French for black) = Nominal, Ordinal, Interval, Ratio, in increasing power.
Each level inherits everything the level below it can do.

---

## 2.3 Data Objects and Attributes

**Data object** = an entity. Also called **samples, examples, instances, data points, objects, tuples**.
- sales database → customers, store items, sales
- medical database → patients, treatments
- university database → students, professors, courses

**Database rows → data objects; columns → attributes.**

**Attribute** (also **dimension, feature, variable**) = a data field representing a characteristic or
feature of a data object. E.g. customer_ID, name, address.

### Attribute types (the taught taxonomy)

```
Attribute
├── Nominal ......... categories, states, "names of things"
│                     Hair_color = {auburn, black, blond, brown, grey, red, white}
├── Binary .......... a nominal attribute with only 2 states (0 and 1)
│   ├── Symmetric ... both outcomes equally important — e.g. gender
│   └── Asymmetric .. outcomes NOT equally important — e.g. a medical test (positive vs negative)
│                     Convention: assign 1 to the MOST IMPORTANT outcome (e.g. HIV positive)
├── Ordinal ......... meaningful order/ranking, but the magnitude between successive
│                     values is NOT known.  Size = {small, medium, large}, grades, army rankings
└── Numeric (quantitative, integer or real-valued)
    ├── Interval-scaled ... equal-sized units, ordered, NO true zero  (°C, °F, calendar dates)
    └── Ratio-scaled ...... inherent zero (Kelvin, length, counts, monetary quantities)
```

### Discrete vs Continuous

| | Discrete Attribute | Continuous Attribute |
|---|---|---|
| Values | **Finite** set of values | **Real numbers** |
| Examples | zip codes, profession, set of words in a document collection | temperature, height, weight |
| Representation | often integer variables | **floating-point** variables |
| Note | **Binary attributes are a special case of discrete attributes** | Practically, real values can only be measured/represented with a finite number of digits |

---

## 2.4 Important Characteristics of Structured Data (4 — memorise with their one-liners)

| Characteristic | One-line hook | Detail |
|---|---|---|
| **Dimensionality** | *curse of dimensionality* | The **number of attributes** the objects possess. Analysing low-dimensional data is *qualitatively* different from moderate/high dimensional. Motivates **dimensionality reduction** in preprocessing. |
| **Sparsity** | *only presence counts* | A special case of skewed data. For sparse binary/count/continuous data, **most attributes of an object are 0** — often **fewer than 1% non-zero**. Sparsity is an **advantage**: only non-zero values need storing and manipulating → big savings in computation time and storage. |
| **Resolution** | *patterns depend on the scale* | Properties of data differ at different resolutions. *Earth's surface looks uneven at a few metres but smooth at tens of kilometres.* **If resolution is too fine the pattern may be buried in noise; too coarse and the pattern disappears.** Atmospheric pressure over *hours* shows storms; over *months* such phenomena are undetectable. |
| **Distribution** | *centrality and dispersion* | The **frequency of occurrence** of various values/sets of values for the attributes. Equivalently, a description of the **concentration of objects in various regions of the data space**. |

---

## 2.5 Types of Data Sets

| Category | Members |
|---|---|
| **Record** | Relational records · **Data matrix** (numerical matrix, crosstabs) · **Document data** (text documents as a **term-frequency vector**) · **Transaction data** |
| **Graph and network** | World Wide Web · social/information networks · molecular structures |
| **Ordered** | **Video** data (sequence of images) · **Temporal** data (time series) · **Sequential** data (transaction sequences) · **Genetic sequence** data |
| **Spatial, image and multimedia** | Spatial data (maps) · image data · video data |

---

## 2.6 DATA QUALITY — the four problems

The framing questions from the slide (use them to structure any long answer):
1. What kinds of data quality problems (exist)?
2. How can we **detect** problems with the data?
3. What can we **do** about these problems?

| Problem | Definition (as taught) | Notes |
|---|---|---|
| **Noise** | **Modification of original values** | Examples: distortion of a person's voice on a poor phone line; "snow" on a television screen. Slide illustration: *Two Sine Waves* vs *Two Sine Waves + Noise*. |
| **Outliers** | Data objects with **characteristics considerably different from most of the other data objects** in the data set | May be errors *or* genuinely interesting (fraud, intrusion) — see 2.7. |
| **Missing values** | Values not recorded | See causes and handling below. |
| **Duplicate data** | Objects that are duplicates, or **almost duplicates**, of one another | **Major issue when merging data from heterogeneous sources.** Example: the same person with multiple email addresses. The process of dealing with them = **data cleaning**. |

### Missing values — reasons (2) and handling (4) ← classic 4-mark question

**Reasons:**
1. **Information is not collected** — e.g. people decline to give their age and weight.
2. **Attributes may not be applicable to all cases** — e.g. annual income is not applicable to children.

**Handling strategies:**
1. **Eliminate data objects** (delete the rows — or the attribute).
2. **Estimate missing values** (imputation: mean / median / mode / model-based).
3. **Ignore the missing value during analysis** (algorithms that tolerate NaN).
4. **Replace with all possible values, weighted by their probabilities** (probabilistic/multiple imputation).

### ★ Noise vs Outlier (very commonly confused — know the difference)
- **Noise** = *random error/variance in a measured variable*; it distorts existing values. Noise is
  generally **not interesting** and should be reduced.
- **Outlier** = a *legitimate data object* whose characteristics differ markedly from the rest. It
  may be an error **or** the most important signal in the dataset (fraud, intrusion, disease).

### ★ Outlier vs Error (this exact question is in your Activity-1 brief)
- An **outlier** is *unusual but possible* — e.g. hours-per-week = 99 for a workaholic entrepreneur.
- An **error** is *impossible or contradicts the data's definition* — e.g. age = −5, age = 250,
  income = −20,000, BMI = 204.
- **Rule: delete/repair errors; investigate, cap, transform or retain outliers — never delete an
  outlier merely because it is extreme.** Deleting real extremes biases the population you represent.

---

## 2.7 DATA PREPROCESSING — the 7 techniques (list them in this order)

1. **Aggregation**
2. **Sampling**
3. **Dimensionality Reduction**
4. **Feature subset selection**
5. **Feature creation**
6. **Discretization and Binarization**
7. **Attribute Transformation**

---

### (1) Aggregation — *"Sometimes less is more"*

**Combining two or more attributes (or objects) into a single attribute (or object).**

**Purpose — three reasons:**
- **Data reduction** — reduce the number of attributes or objects.
- **Change of scale** — cities aggregated into regions, states, countries.
- **More "stable" data** — **aggregated data tends to have less variability.**

Slide evidence: *Variation of Precipitation in Australia* — the standard deviation of **average
yearly** precipitation is much smaller/tighter than that of **average monthly** precipitation.
That histogram pair is the proof of "aggregation reduces variability".

---

### (2) Sampling

> **Sampling is the main technique employed for data selection.** Used for both the preliminary
> investigation of the data and the final data analysis.

**Why statisticians sample:** obtaining the entire set of data of interest is **too expensive or
time consuming**.
**Why data mining samples:** *processing* the entire set of data of interest is too expensive or
time consuming. *(Note the subtle difference — it's a favourite exam distinction.)*

**The key principle for effective sampling:**
> *Using a sample will work almost as well as using the entire data set — **if the sample is
> representative**. A sample is representative if it has approximately the same property (of
> interest) as the original set of data.*

**Types of sampling:**

| Type | Definition |
|---|---|
| **Simple Random Sampling** | There is an **equal probability** of selecting any particular item |
| — *without replacement* | As each item is selected, it is **removed** from the population |
| — *with replacement* | Objects are **not removed** as they are selected → **the same object can be picked more than once** |
| **Stratified sampling** | **Split the data into several partitions, then draw random samples from each partition** |

**Sample size matters** — the slide shows the same 2-D point cloud at **8000 → 2000 → 500 points**.
At 8000 the structure/clusters are obvious; at 500 some structure is lost. *Too small a sample
loses the pattern; too large loses the efficiency benefit.*

> Use **stratified sampling** whenever the target is imbalanced (e.g. Adult dataset 76% vs 24%) so
> the rare class is properly represented.

---

### (3) Dimensionality Reduction

**Motivation:** data sets can have a huge number of features. *Example from the slide:* a set of
documents where each document is a vector of word frequencies → **thousands or tens of thousands of
attributes, one per vocabulary word.**

**The Curse of Dimensionality** (define it exactly like this):
> The phenomenon that **many types of data analysis become significantly harder as the
> dimensionality of the data increases.** Specifically, **as dimensionality increases, the data
> becomes increasingly sparse in the space that it occupies.** Thus the data objects we observe are
> quite possibly **not a representative sample** of all possible objects.
> - **For classification:** there are not enough data objects to build a model that reliably assigns
>   a class to all possible objects.
> - **For clustering:** the differences in **density** and in the **distances between points** —
>   which are critical for clustering — **become less meaningful**.

**Purpose of dimensionality reduction (4):**
1. Avoid the curse of dimensionality
2. Reduce the **time and memory** required by data mining algorithms
3. Allow data to be **more easily visualized**
4. May help **eliminate irrelevant features or reduce noise**

**Techniques:** **Principal Component Analysis (PCA)**, **Singular Value Decomposition (SVD)**,
plus supervised and non-linear techniques.

**PCA (as defined on the slide):**
> *Goal is to find a **projection that captures the largest amount of variation** in the data.*
> *PCA is a dimensionality-reduction method used to reduce the dimensionality of large data sets by
> **transforming a large set of variables into a smaller one that still contains most of the
> information** in the large set.*

The slide picture: 2-D cloud of points with axes x₁, x₂ and a vector **e** drawn along the direction
of maximum spread — **e is the first principal component**.

---

### (4) Feature Subset Selection — *another way to reduce dimensionality*

**Two kinds of features you can remove:**

| | Definition | Slide example |
|---|---|---|
| **Redundant features** | **Duplicate much or all of the information** contained in one or more other attributes | **Purchase price of a product and the amount of sales tax paid** |
| **Irrelevant features** | Contain **no information useful for the data mining task at hand** | **Students' ID is irrelevant to predicting students' GPA** |

**Four techniques (memorise all four names + one line each):**

| Approach | How it works |
|---|---|
| **Brute-force** | Try **all possible feature subsets** as input to the data mining algorithm |
| **Embedded** | Feature selection occurs **naturally as part of** the data mining algorithm (e.g. decision trees, Lasso) |
| **Filter** | Features are selected **before** the data mining algorithm is run (e.g. correlation, chi-square) |
| **Wrapper** | Use the data mining algorithm **as a black box** to find the best subset of attributes (e.g. RFE) |

Mnemonic: **B-E-F-W** — or remember the order of *coupling* to the model: Filter (none) →
Wrapper (uses model as black box) → Embedded (inside the model) → Brute force (exhaustive).

---

### (5) Feature Creation / Mapping Data to a New Space
Create new attributes that capture the important information far more efficiently.
**Mapping data to a new space:** the slide shows *Two Sine Waves* + noise in the time domain being
transformed by a **Fourier transform** into the **frequency** domain, where the two frequencies show
up as two clean spikes — the pattern hidden by noise in one space is obvious in another.
Also mentioned: **Wavelet transform**.

---

### (6) Discretization and Binarization
- **Discretization** — convert a continuous attribute into ordinal/categorical bins (e.g. age → teen/adult/mid/senior).
- **Binarization** — convert a categorical/continuous attribute into one or more binary (0/1) attributes.

---

### (7) Attribute Transformation

> **A function that maps the entire set of values of a given attribute to a new set of replacement
> values, such that each old value can be identified with one of the new values.**

- **Simple functions:** xᵏ, log(x), eˣ, |x|
- **Standardization and Normalization**

**From the instructor's own speaker notes (learn this wording — he wrote it himself):**
> **Standardization:** *"In statistics it refers to subtracting off the means and dividing by the
> standard deviation (to make an entire set of values have a particular property, e.g. mean 0 and SD 1)."*
> **Normalization:** *"It refers to various techniques to adjust to differences among attributes in
> terms of frequency of occurrence, mean, variance, range etc."*
> *"Normalization and standardization terms are normally used interchangeably."*

Practical formulas (know both):
- **Min–max normalization:** x' = (x − min) / (max − min) → range [0, 1]
- **Z-score standardization:** z = (x − μ) / σ → mean 0, SD 1

---

## 2.8 Similarity and Dissimilarity

Important for a number of data science techniques:
- **Clustering**
- **Nearest neighbor classification**
- **Anomaly detection**

> *In some cases, the original dataset is **not needed** once similarity/dissimilarity has been
> computed — this means **transforming the data to a similarity (dissimilarity) space** and then
> performing the analysis there.*

Quick reference (useful if he asks for measures):
- **Euclidean distance** (numeric), **Manhattan**, **Minkowski**
- **Cosine similarity** (documents / term-frequency vectors)
- **Jaccard coefficient** (asymmetric binary / sets)
- **Simple Matching Coefficient** (symmetric binary)
- Similarity s and dissimilarity d often relate as **s = 1 − d** for normalized measures.
