# 03 — NUMERICALS: Every calculation type in Lec 9–17

Every answer below was computed in Python and checked. Each topic has: **the formula → a worked example (step by step) → a practice question with answer at the bottom of the file.**

**Exam method for any numerical (write these steps, they earn method marks even if you slip on arithmetic):**
1. Write the formula.
2. List the given values with symbols.
3. Substitute.
4. Compute, keeping 3–4 decimals.
5. Write a one-line interpretation ("so 'prediction' is more distinctive than 'machine'").

**Calculator notes:** log₁₀ is the `log` key; natural log is `ln`. log₁₀(1)=0, log₁₀(2)=0.3010, log₁₀(3)=0.4771, log₁₀(1.5)=0.1761, log₁₀(4/3)=0.1249, ln(4/3)=0.2877, ln(2)=0.6931, ln(1.5)=0.4055, ln(2.5)=0.9163, ln(5/3)=0.5108.

---

## Topic map (what can come, from which lecture)

| # | Numerical type | Lecture | Likelihood |
|---|---|---|---|
| 1 | TF-IDF by hand (3 formula versions) | 14, 15 | ★★★ |
| 2 | Cosine similarity | 14 (repos) | ★★★ |
| 3 | Matrix shape / sparsity | 15 (Lab 15) | ★★ |
| 4 | Confusion matrix metrics + F1 | 11 | ★★★ |
| 5 | Bias², variance, expected error | 11 | ★★ |
| 6 | Loss functions / MSE | 10 | ★★ |
| 7 | k-fold CV mean ± SE, split sizes | 10 | ★★ |
| 8 | Outliers: z-score and IQR fence | 12 | ★★★ |
| 9 | SMOTE interpolation & counts | 12 | ★★★ |
| 10 | Imbalance ratio, resampling counts | 12 | ★★ |
| 11 | Sample size + stratified allocation | 12 | ★★ |
| 12 | HDFS blocks & replication | 16 | ★★★ |
| 13 | MapReduce trace | 16, 17 | ★★★ |
| 14 | Big-data timing, grid-search counts | 16, 12 | ★ |

---

## 1. TF-IDF BY HAND ★★★

There are **three versions** in your course. Read the question to see which one it wants; if it doesn't say, use the Lec 15 slide formula and **state the log base**.

| Version | IDF formula | Where |
|---|---|---|
| **A. Basic (Lec 15 slide)** | IDF = log(N / DF) | Lec 15, Lab 15 README |
| **B. Smoothed (Lec 14 worked example)** | IDF = log₁₀[(1 + N) / (1 + DF)] | Lec 14 |
| **C. scikit-learn** | IDF = ln[(1 + N)/(1 + DF)] **+ 1**, then **L2-normalize** each row | Lec 14 note |

In all three: **TF-IDF(t, d) = TF(t, d) × IDF(t)**, where TF = raw count of t in d, DF = number of documents containing t, N = number of documents.

### Worked example 1 — the Lec 14 slide example (Version B)
D1 = "machine learning improves prediction", D2 = "machine learning analyzes data", D3 = "machine learning supports healthcare". N = 3.

| Term | TF in D1 | DF | IDF = log₁₀[(1+3)/(1+DF)] | TF-IDF |
|---|---|---|---|---|
| machine | 1 | 3 | log₁₀(4/4) = **0** | 1 × 0 = **0.000** |
| prediction | 1 | 1 | log₁₀(4/2) = log₁₀ 2 = **0.3010** | 1 × 0.3010 = **0.301** |

Interpretation: "machine" is in every document, so it says nothing about D1. "prediction" appears only in D1, so it is distinctive.

The same corpus in **sklearn (Version C)**: idf(machine) = ln(4/4)+1 = **1.0**, idf(prediction) = ln(4/2)+1 = **1.6931**. After L2 normalization D1 = machine 0.36, learning 0.36, improves 0.609, prediction 0.609. (Verified.) Notice sklearn **never gives 0** to a common word — the "+1" keeps it above zero.

### Worked example 2 — all three versions on one corpus
D1 = "apple banana apple", D2 = "banana cherry", D3 = "cherry date apple". N = 3.

**Step 1 — DF table**

| Term | D1 | D2 | D3 | DF |
|---|---|---|---|---|
| apple | 2 | 0 | 1 | 2 |
| banana | 1 | 1 | 0 | 2 |
| cherry | 0 | 1 | 1 | 2 |
| date | 0 | 0 | 1 | 1 |

**Step 2 — IDF (three ways)**

| Term | A: log₁₀(3/DF) | B: log₁₀(4/(1+DF)) | C: ln(4/(1+DF)) + 1 |
|---|---|---|---|
| apple, banana, cherry (DF=2) | log₁₀ 1.5 = **0.1761** | log₁₀(4/3) = **0.1249** | 0.2877 + 1 = **1.2877** |
| date (DF=1) | log₁₀ 3 = **0.4771** | log₁₀ 2 = **0.3010** | 0.6931 + 1 = **1.6931** |

**Step 3 — TF-IDF for D1** (TF: apple = 2, banana = 1)

| | apple | banana | cherry | date |
|---|---|---|---|---|
| A | 2 × 0.1761 = **0.3522** | 0.1761 | 0 | 0 |
| B | 2 × 0.1249 = **0.2499** | 0.1249 | 0 | 0 |
| C (raw) | 2 × 1.2877 = 2.5754 | 1.2877 | 0 | 0 |

**Step 4 — L2 normalization (Version C only)**
‖v‖ = √(2.5754² + 1.2877²) = √(6.633 + 1.658) = √8.291 = 2.8794
apple = 2.5754 / 2.8794 = **0.8944**, banana = 1.2877 / 2.8794 = **0.4472** ✓ (matches sklearn exactly)

Shortcut: when two terms share an IDF, normalization just normalizes the counts. (2, 1) / √5 = (0.894, 0.447).

### Things examiners ask about TF-IDF numbers
- A word in **every** document: Version A/B → IDF = **0** → TF-IDF = 0. Version C → IDF = **1**.
- If a question says "TF = count / total words in the document", use that instead: TF(apple, D1) = 2/3 → TF-IDF_A = 0.667 × 0.1761 = **0.1174**.
- Always say which log you used. log₁₀ and ln give different numbers; both are fine if stated.

### Practice 1 ☐
D1 = "data science uses data", D2 = "data mining finds patterns", D3 = "science explains patterns", D4 = "mining gold". N = 4. Using **Version A with log₁₀**, compute TF-IDF in D1 of **data**, **science**, **uses**. Then say which word best describes D1.

---

## 2. COSINE SIMILARITY ★★★

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\;\|\mathbf{b}\|} = \frac{\sum a_i b_i}{\sqrt{\sum a_i^2}\,\sqrt{\sum b_i^2}}$$

Range for TF-IDF vectors: **0** (no shared words) to **1** (same direction). TF-IDF is never negative, so cosine is never negative.

### Worked example
a = (1, 2, 0, 1), b = (2, 1, 1, 0)
- Dot product: 1·2 + 2·1 + 0·1 + 1·0 = **4**
- ‖a‖ = √(1+4+0+1) = √6 = 2.449; ‖b‖ = √(4+1+1+0) = √6 = 2.449
- cos = 4 / (2.449 × 2.449) = 4 / 6 = **0.667**

**Shortcut for sklearn TF-IDF rows:** they already have length 1, so **cosine = just the dot product**. That's why `cosine_similarity` on TF-IDF is cheap.

### Practice 2 ☐
Query q = (1, 0, 2), document d = (2, 1, 2). Find cos(q, d).

---

## 3. MATRIX SHAPE AND SPARSITY ★★

- TF-IDF/BoW matrix shape = **(number of documents, vocabulary size)**.
- `cosine_similarity(X, X)` shape = **(n_docs, n_docs)**. `cosine_similarity(query, X)` = **(1, n_docs)**.
- **Sparsity % = 100 × (1 − non-zeros / (rows × cols))**. In code: `100 * (1 - X.nnz / (X.shape[0]*X.shape[1]))`.

### Worked example
The 4-document corpus in Practice 1 has 8 unique terms. Count matrix:
```
         data explains finds gold mining patterns science uses
D1         2     0       0    0     0      0        1     1
D2         1     0       1    0     1      1        0     0
D3         0     1       0    0     0      1        1     0
D4         0     0       0    1     1      0        0     0
```
Non-zeros = 3 + 4 + 3 + 2 = **12**. Total = 4 × 8 = 32. Sparsity = 100 × (1 − 12/32) = **62.5%** ✓ (verified).

Real corpora are far sparser: the 10-document class dataset is **88.5%** zeros with 537 terms.

---

## 4. CONFUSION MATRIX METRICS ★★★

| | Actual + | Actual − |
|---|---|---|
| **Predicted +** | TP | FP (Type I) |
| **Predicted −** | FN (Type II) | TN |

| Metric | Formula | One-line meaning |
|---|---|---|
| Accuracy | (TP+TN) / all | fraction correct |
| Precision (PPV) | TP / (TP+FP) | of my "yes", how many right |
| Recall = Sensitivity = TPR | TP / (TP+FN) | of real "yes", how many found |
| Specificity = TNR | TN / (TN+FP) | of real "no", how many cleared |
| NPV | TN / (TN+FN) | of my "no", how many right |
| FPR | FP / (FP+TN) = 1 − Specificity | false alarm rate |
| FNR | FN / (TP+FN) = 1 − Recall | miss rate |
| F1 | 2PR / (P+R) | harmonic mean of P and R |

### Worked example — building the matrix from a word problem (the hard part)
*"1000 emails, 80 are spam. The filter flags 90 emails as spam; 60 of the flagged ones are really spam."*

Fill in this order — always start from the sentence that names TP:
1. TP = 60 (flagged AND spam)
2. FP = flagged − TP = 90 − 60 = **30**
3. FN = actual spam − TP = 80 − 60 = **20**
4. TN = total − TP − FP − FN = 1000 − 60 − 30 − 20 = **890**

| Metric | Calculation | Value |
|---|---|---|
| Accuracy | (60+890)/1000 | **0.950** |
| Precision | 60/90 | **0.667** |
| Recall | 60/80 | **0.750** |
| Specificity | 890/920 | **0.967** |
| F1 | 2(0.667)(0.75)/(0.667+0.75) | **0.706** |

Interpretation: 95% accuracy looks great, but a third of the flagged mail is not spam.

### Worked example — working backwards from metrics
*"500 patients screened, 50 have cancer. Recall = 0.9, precision = 0.6."*
- Recall: TP / 50 = 0.9 → **TP = 45**, so **FN = 5**
- Precision: 45 / (45+FP) = 0.6 → 45+FP = 75 → **FP = 30**
- **TN** = 500 − 45 − 30 − 5 = **420**
- Specificity = 420/450 = **0.933**; Accuracy = 465/500 = **0.93**; F1 = 2(0.6)(0.9)/1.5 = **0.72**

### The accuracy trap (Lec 11 + 12)
"Always predict the majority class" → TP = 0 → **recall = 0**, but accuracy = majority share.
- Lec 11 tweets: 999,900 / 1,000,000 = **99.99%** accuracy, recall 0.
- Lec 12 fraud: 284,315 / 284,807 = **99.83%** accuracy, recall 0.

### Practice 4 ☐
TP = 40, FP = 10, FN = 20, TN = 130. Compute accuracy, precision, recall, specificity, F1, FPR.

---

## 5. BIAS², VARIANCE, EXPECTED ERROR ★★

$$E\big[(Y-\hat f(X))^2\big] = \text{Bias}^2 + \text{Var} + \sigma^2$$

- **Bias² = (mean prediction − true value)²**
- **Variance = average of (each prediction − mean prediction)²**

### Worked example
Same model trained on 5 datasets predicts for one test point: **60, 64, 58, 62, 66**. True value = **70**.
1. Mean prediction = (60+64+58+62+66)/5 = 310/5 = **62**
2. Bias = 62 − 70 = −8 → **Bias² = 64**
3. Deviations: −2, +2, −4, 0, +4 → squares 4, 4, 16, 0, 16 → sum 40 → **Var = 40/5 = 8**
4. Expected error (ignoring noise) = 64 + 8 = **72**

Diagnosis: Bias² (64) dominates variance (8) → **high bias → underfitting** → make the model more complex / add features.

Compare: predictions 90, 50, 110, 70, 80 with truth 75 → mean 80, Bias² = 25, Var = **400** → **high variance → overfitting**.

### Practice 5 ☐
Truth = 50. Predictions: 44, 46, 48, 42, 40. Find Bias², Variance, and say under- or overfitting.

---

## 6. LOSS FUNCTIONS ★★ (Lec 10)

- **0/1 loss** (classification): 1 if wrong, 0 if right. Average 0/1 loss = error rate.
- **Square loss** (regression): (predicted − target)².
- **MSE** = mean of square losses.

### Worked example
- Predicted price 120, actual 100 → square loss = 20² = **400**.
- Lec 11 slide: Y = 300,000, ŷ = 280,000 → (20,000)² = **400,000,000**.
- Actual (3, 5, 2, 8), predicted (2, 5, 4, 6) → errors 1, 0, −2, 2 → squares 1, 0, 4, 4 → MSE = 9/4 = **2.25**.
- 10 predictions, 3 wrong → average 0/1 loss = **0.3** (accuracy 0.7).

Polynomial of degree M has **M + 1 parameters** (w₀ … w_M). M = 9 → 10 parameters. With 10 data points, M = 9 can pass through every point → training error 0 → overfit (Lec 10 slide).

---

## 7. CROSS-VALIDATION AND SPLITS ★★ (Lec 10)

- 80/20 split of 500 rows → **400 train / 100 test**.
- k-fold: each fold has N/k rows; each model trains on N − N/k. 120 rows, k = 5 → folds of 24; train on **96**, test on **24**, 5 models.
- Report **T̂ ± SE**, where T̂ = mean(T₁…T_k), **SE = SD / √k**.

### Worked example
Fold accuracies: 0.78, 0.81, 0.85, 0.80, 0.76
- Mean = 4.00 / 5 = **0.80**
- Deviations: −0.02, 0.01, 0.05, 0, −0.04 → squares 0.0004, 0.0001, 0.0025, 0, 0.0016 → sum 0.0046
- SD (sample, ÷ (k−1) = 4) = √(0.00115) = **0.0339**
- SE = 0.0339 / √5 = 0.0339 / 2.236 = **0.0152**
- Report: **0.80 ± 0.015**

(If you divide by k instead: SD = 0.0303, SE = 0.0136. Either is accepted if you show the formula — state which you used.)

**GridSearch count (Lec 12):** grid `{k: [10,15,20], k_neighbors: [3,5], C: [0.1,1,10]}` → 3 × 2 × 3 = **18** combinations × 5 folds = **90** model fits.

---

## 8. OUTLIERS: Z-SCORE AND IQR FENCE ★★★ (Lec 12)

- **z = (x − μ) / σ**; flag if **|z| > 3**.
- **IQR = Q3 − Q1**; fence = **[Q1 − 1.5·IQR, Q3 + 1.5·IQR]**; outside = outlier.

Quartiles by hand (median-of-halves method): sort; Q2 = median; Q1 = median of the lower half; Q3 = median of the upper half (with an odd count, leave the median out of both halves).

### Worked example 1 — 11 values
Data: 12, 15, 14, 10, 18, 16, 13, 95, 11, 17, 14
1. Sorted: 10, 11, 12, 13, 14, **14**, 15, 16, 17, 18, 95 → median = 14
2. Lower half: 10, 11, **12**, 13, 14 → **Q1 = 12**; upper half: 15, 16, **17**, 18, 95 → **Q3 = 17**
3. IQR = 17 − 12 = **5**
4. Fence = [12 − 7.5, 17 + 7.5] = **[4.5, 24.5]** → **95 is an outlier**
5. z of 95: μ = 21.36, σ = 23.40 → z = (95 − 21.36)/23.40 = **3.15** → also flagged

### Worked example 2 — where z-score FAILS (exam favourite)
Data: 2, 4, 5, 6, 7, 8, 9, 30
- Lower half 2, 4, 5, 6 → Q1 = 4.5; upper half 7, 8, 9, 30 → Q3 = 8.5; IQR = **4**
- Fence = [4.5 − 6, 8.5 + 6] = **[−1.5, 14.5]** → **30 is an outlier**
- But z(30): μ = 8.875, σ = 8.25 → z = **2.56 < 3** → **z-score misses it**

Why: the outlier inflates σ, which shrinks its own z-score (**masking**). IQR uses quartiles, which the outlier barely moves, so IQR is **robust**. Same reason you prefer **median** imputation over mean.

---

## 9. SMOTE ★★★ (Lec 12)

$$x_{new} = x_i + \lambda\,(\hat x - x_i), \quad \lambda \sim U(0,1)$$
$\hat x$ = a randomly chosen one of the **k nearest minority** neighbours of $x_i$.

### Worked example (the slide's)
x_i = (2, 3), neighbours (4, 7), (3, 2), (1, 4); chosen x̂ = (4, 7); λ = 0.25
- Difference: (4−2, 7−3) = (2, 4)
- λ × difference = (0.5, 1.0)
- x_new = (2 + 0.5, 3 + 1.0) = **(2.5, 4.0)**

### More
- x_i = (3, 8), x̂ = (7, 2), λ = 0.75 → diff (4, −6) → 0.75 × diff = (3, −4.5) → x_new = **(6, 3.5)**
- x_i = (1, 4), x̂ = (5, 2), λ = 0.4 → diff (4, −2) → (1.6, −0.8) → **(2.6, 3.2)**
- λ = 0 → x_new = x_i; λ = 1 → x_new = x̂. The new point **always lies on the segment between them** (can't go outside the minority region).

**How many synthetic points?** N_new = **r · N_maj − N_min**
- 9,500 majority, 500 minority, target r = 1 → 9,500 − 500 = **9,000** (matches the slide's `n_new=9000`)
- r = 0.5 → 4,750 − 500 = **4,250**

---

## 10. IMBALANCE RATIO AND RESAMPLING COUNTS ★★ (Lec 12)

- **IR = N_majority / N_minority**. Fraud: 284,315 / 492 ≈ **578 : 1**. Fraud share = 492/284,807 = **0.172%**.
- `RandomOverSampler(sampling_strategy=0.5)` on {0: 9500, 1: 500} → minority grows to 0.5 × 9500 = **4750** → {0: 9500, 1: 4750}.
- `RandomUnderSampler(sampling_strategy="majority")` → majority shrinks to minority size → {0: **500**, 1: 500}.
- Decision guide: IR < ~10 (mild) → try class weights + threshold first.

---

## 11. SAMPLE SIZE AND STRATIFIED ALLOCATION ★★ (Lec 12)

### Sample size for a proportion
$$n_0 = \frac{z^2\,p(1-p)}{e^2}, \qquad n = \frac{n_0}{1 + (n_0-1)/N}\ \text{(finite population correction)}$$
z = 1.96 for 95% confidence. Unknown p → use **p = 0.5** (the safest/largest).

**Worked:** 95%, p = 0.5, e = 0.05 → n₀ = (3.8416 × 0.25)/0.0025 = **384.16 → 385** (always round **up**).
With N = 2000: n = 384.16 / (1 + 383.16/2000) = 384.16 / 1.1916 = **322.4 → 323**.
e = 0.03 → n₀ = 0.9604/0.0009 = **1067.1 → 1068**.

**Error ∝ 1/√n:** to **halve** the error you need **4×** the data; to cut it to a third, 9×.

### Stratified allocation
- **Proportional:** n_h = n · N_h / N
- **Neyman:** n_h = n · N_h σ_h / Σ(N_k σ_k) → more samples where the stratum is big **and** variable.

**Slide example:** n = 100; Healthy N = 9000, σ = 1; Diseased N = 1000, σ = 3.
Proportional: 90 / 10. Neyman weights 9000·1 : 1000·3 = 9000 : 3000 = 3 : 1 → **75 / 25**.

**Practice-style:** n = 300; A: N = 6000, σ = 2; B: N = 4000, σ = 6.
Proportional: 300 × 0.6 = **180**, 300 × 0.4 = **120**.
Neyman: weights 12,000 : 24,000 = 1 : 2 → **100 / 200**.

---

## 12. HDFS BLOCKS AND REPLICATION ★★★ (Lec 16 + repo)

Defaults: **block = 128 MB** (Hadoop 2.x+; the original GFS/MapReduce paper used **64 MB**), **replication = 3**.

- Blocks = ⌈file size / block size⌉ (round **up**)
- Replicas stored = blocks × replication
- Raw disk used ≈ file size × replication (the last block only uses its actual size, not a full 128 MB)

| File | Blocks | Last block | Replicas (×3) | Disk used |
|---|---|---|---|---|
| 1 GB (1024 MB) | 1024/128 = **8** | full | **24** | **3 GB** |
| 300 MB | ⌈2.34⌉ = **3** (128 + 128 + **44**) | 44 MB | **9** | **900 MB** |
| 500 MB | ⌈3.91⌉ = **4** (3×128 + **116**) | 116 MB | **12** | **1500 MB** |
| 2 GB | **16** | full | **48** | **6 GB** |

Map tasks: one input split per block → a 1 GB file gives **8 map tasks**.

Replica placement (rack-aware): 1st on the writer's node, 2nd on a **different rack**, 3rd on the **same rack as the 2nd** but another node.

---

## 13. MAPREDUCE TRACE ★★★ (Lec 16, 17, repo)

Write it as three stages: **Map output → Shuffle & sort (group by key) → Reduce output.**

### Worked example (repo doc)
Input: `"the cat sat"`, `"the dog ran"`

| Stage | Contents |
|---|---|
| Map | (the,1) (cat,1) (sat,1) (the,1) (dog,1) (ran,1) |
| Shuffle & sort | cat:[1] dog:[1] ran:[1] sat:[1] the:[1,1] |
| Reduce | (cat,1) (dog,1) (ran,1) (sat,1) (the,2) |

### Lec 16 diagram example
After shuffle: a: [1, 5], b: [2, 7], c: [2, 9, 8] → reduce (sum) → **a 6, b 9, c 19**.

### Lec 16/17 slide input (verified)
```
hadoop is a framework written in java
hadoop supports parallel processing
and is a simple framework
```
Map emits **16** pairs (7 + 4 + 5 words). 12 distinct keys → reducer output:
`a 2, and 1, framework 2, hadoop 2, in 1, is 2, java 1, parallel 1, processing 1, simple 1, supports 1, written 1`

### Counting questions
- Number of mapper output lines = **total words** (every occurrence emits a pair).
- Number of reducer output lines = **number of distinct words**.
- With a **combiner** (local sum before shuffle), mapper 1 sends one pair per distinct word *in its split* → less network traffic. Valid for sum/count/max, **not for median**.
- Inverted index mapper emits each word **once per document** (because of `set()`), so "big big data" in d1 emits 2 pairs, not 3.

---

## 14. BIG-DATA TIMING AND MISC ★ (Lec 16)

- Web: 20 billion pages × 20 KB = **400 TB**.
- Read at 35 MB/s: 400 × 10⁶ MB / 35 MB/s = 11.4 million s = **132 days ≈ 4 months** (slide: "~four months").
- 1000 machines in parallel: 11.4 million / 1000 = 11,429 s ≈ **3.2 hours** (slide: "< 3 hours" with rounding).
- 12-hour flight → **84 TB** of sensor data. Facebook daily data **30+ PB** > all US academic libraries (2 PB).

---

## ANSWERS TO PRACTICE

**Practice 1** — N = 4, log₁₀.
- data: TF = 2, DF = 2 → IDF = log₁₀(4/2) = 0.3010 → **0.602**
- science: TF = 1, DF = 2 → **0.301**
- uses: TF = 1, DF = 1 → IDF = log₁₀ 4 = 0.6021 → **0.602**
- "data" and "uses" tie at 0.602 — "data" because it's frequent in D1, "uses" because it's unique to D1. "science" is lower (in two documents, only once in D1).
- (In sklearn, D1 normalized = data 0.778, science 0.389, uses 0.493 — there sklearn's "+1" makes the repeated "data" win.)

**Practice 2** — dot = 2 + 0 + 4 = 6; ‖q‖ = √5 = 2.236; ‖d‖ = √9 = 3 → cos = 6/6.708 = **0.894**

**Practice 4** — Total 200. Accuracy (40+130)/200 = **0.85**; Precision 40/50 = **0.80**; Recall 40/60 = **0.667**; Specificity 130/140 = **0.929**; F1 = 2(0.8)(0.667)/1.467 = **0.727**; FPR = 10/140 = **0.071**

**Practice 5** — Mean = 220/5 = 44 → Bias² = (44−50)² = **36**; deviations 0, 2, 4, −2, −4 → squares 0, 4, 16, 4, 16 = 40 → Var = **8** → bias dominates → **underfitting**.
