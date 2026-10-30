# MOCK MID EXAM — PAPER B — SOLUTIONS & MARKING
All numericals and code outputs verified by running them.

---

## SECTION A — MCQs

| Q | Ans | Why |
|---|---|---|
| A1 | **b** | CV estimates test error (Lec 11: "do a cross-validation to estimate the test error"); it improves models only indirectly via selection. |
| A2 | **b** | Lec 11 slide: M = 9 with more data (N = 15) → "pretty good – slightly overfit?". More data tames complexity. |
| A3 | **c** | w's are parameters learned; M is set by the data scientist. |
| A4 | **b** | Lec 10 leakage example. |
| A5 | **a** | TPR / FPR. |
| A6 | **c** | TNR = TN/(TN+FP). |
| A7 | **b** | Missing indicator keeps the "was missing" signal (Lec 12 table). |
| A8 | **b** | ADASYN focuses on hard-to-learn minority regions (can also boost noise). |
| A9 | **c** | Decision guide: mild IR < 10 → class weights + tuned threshold first. |
| A10 | **b** | Lec 12 aggregation slide. |
| A11 | **b** | n_h ∝ N_h σ_h. |
| A12 | **b** | Linguistic, high accuracy, slower; WordNet. |
| A13 | **b** | TfidfVectorizer does not stem. |
| A14 | **b** | (n_queries, n_docs) = (1, 10). |
| A15 | **c** | `vec_loaded.transform([...])` reuses the saved vocabulary/IDF. |
| A16 | **b** | HA: Standby NameNode with shared edit logs (QJM). |
| A17 | **a** | Each file's metadata lives in NameNode memory. |
| A18 | **b** | Map outputs are on the failed node's local disk → lost → re-run. Reduce output in HDFS survives. |
| A19 | **b** | Lec 17 streams slide. |
| A20 | **d** | Structured ≈ 5%, semi-structured 5–10%, unstructured ≈ 80%. |

---

## SECTION B — NUMERICALS

### B1 (7)
**(a)** N = 4. data = **3** (D1, D2, D4) · spark = **2** · hadoop = **2** · streams, fast, stores, beats, everywhere = **1** each.

**(b)** D1 terms (each TF = 1):
- spark: log₁₀(4/2) = **0.3010**
- streams: log₁₀(4/1) = **0.6021**
- data: log₁₀(4/3) = **0.1249**
- fast: log₁₀(4/1) = **0.6021**

Most distinctive: **streams** and **fast** (tie, unique to D1). "data" is weakest (in 3 of 4 docs).

**(c)** **0** — D3 {spark, beats, hadoop} and D4 {data, everywhere} share no words, so the dot product is 0. (Verified: cos(D3, D4) = 0.000.)

**(d)** ln((1+4)/(1+3)) + 1 = ln(1.25) + 1 = 0.2231 + 1 = **1.2231**.

### B2 (7)
**(a)** Diseased = 200 → TP = 0.85 × 200 = **170**, FN = **30**. Healthy = 800 → TN = 0.90 × 800 = **720**, FP = **80**.
**(b)**
- Precision = 170/250 = **0.68**
- Accuracy = (170 + 720)/1000 = **0.89**
- F1 = 2(0.68)(0.85)/(1.53) = **0.756**
- NPV = 720/750 = **0.96**

**(c)** Sensitivity is P(test + | disease). The share of positive results that are real is **precision = 0.68** — only 68%, because the 80 false positives come from the much larger healthy group.

### B3 (5)
**R:** mean = 250/5 = **50** → Bias² = (50 − 60)² = **100**; deviations 2, −2, 1, −1, 0 → squares 4, 4, 1, 1, 0 = 10 → Var = **2**. Total ≈ **102**. High bias, low variance → **underfitting**.
**S:** mean = 300/5 = **60** → Bias² = **0**; deviations −30, 30, −5, 15, −10 → squares 900, 900, 25, 225, 100 = 2150 → Var = **430**. Total ≈ **430**. High variance → **overfitting**.

### B4 (4)
**(a)** Sorted (n = 11): 2, 22, 25, 27, 28, **30**, 31, 33, 35, 36, 100. Median = 30.
Lower half 2, 22, **25**, 27, 28 → Q1 = **25**; upper half 31, 33, **35**, 36, 100 → Q3 = **35**; IQR = **10**.
Fence = [25 − 15, 35 + 15] = **[10, 50]** → outliers: **2 and 100**.
**(b)** z(100) = (100 − 33.55)/22.80 = **2.91**; z(2) = (2 − 33.55)/22.80 = **−1.38**. **Neither** passes |z| > 3 — both outliers inflate σ (masking). IQR catches both.

### B5 (4)
**(a)** Mean = 3.60/5 = **0.72**. Deviations −0.02, 0.02, −0.04, 0, 0.04 → squares 0.0004, 0.0004, 0.0016, 0, 0.0016 = 0.0040. Sample SD = √(0.0040/4) = **0.0316**. SE = 0.0316/√5 = **0.0141**. Report **0.72 ± 0.014**.
**(b)** 4 × 3 × 2 = 24 combinations × 10 folds = **240 fits**.
**(c)** Fold size 24 → each model trains on **96** rows.

### B6 (4)
**(a)** n₀ = 2.576² × 0.25 / 0.04² = 6.6358 × 0.25 / 0.0016 = **1036.8 → 1037**. With N = 1500: n = 1036.84 / (1 + 1035.84/1500) = 1036.84 / 1.6906 = **613.3 → 614**.
**(b)** N = 10,000.
- Proportional: X = 230 × 0.5 = **115**, Y = 230 × 0.3 = **69**, Z = 230 × 0.2 = **46**.
- Neyman weights: 5000·4 = 20,000; 3000·2 = 6,000; 2000·10 = 20,000; total 46,000 → X = 230 × 20/46 = **100**, Y = 230 × 6/46 = **30**, Z = **100**.
Z is the smallest stratum but gets the same share as X because it's the most variable.

### B7 (4)
**(a)** Blocks: 130 MB → 2 (128 + 2), 50 MB → 1, 1 GB → 8 → **11 blocks**. Replication 2 → **22 replicas**. Raw disk ≈ (130 + 50 + 1024) × 2 = **2,408 MB**. (The 50 MB file and the 2 MB tail don't occupy full 128 MB blocks, but each still costs NameNode memory.)
**(b)** x̂ − x_i = (2.0, −1.0, −4.0); × 0.6 = (1.2, −0.6, −2.4); x_new = **(2.7, 1.4, 3.6)**.
**(c)** Without combiner: 3 + 4 = **7 pairs**. With combiner: mapper 1 → (red,2),(blue,1); mapper 2 → (blue,2),(green,1),(red,1) → **5 pairs**. Output: **blue 3, green 1, red 3** (verified).

---

## SECTION C — CODING

### C1 (9)
```python
# mapper.py
#!/usr/bin/env python3
import sys
for line in sys.stdin:
    doc_id, _, text = line.rstrip("\n").partition("\t")
    if not doc_id:
        continue
    for word in set(text.lower().split()):
        print(f"{word}\t{doc_id}")
```
```python
# reducer.py
#!/usr/bin/env python3
import sys

def flush(word, docs):
    if word is not None:
        print(f"{word}\t{','.join(sorted(set(docs)))}")

current, docs = None, []
for line in sys.stdin:
    word, _, doc_id = line.rstrip("\n").partition("\t")
    if word != current:
        flush(current, docs)
        current, docs = word, []
    docs.append(doc_id)
flush(current, docs)
```
**Output (verified):**
```
and     d3
fast    d1
hadoop  d2,d3
is      d1,d2
slow    d2
spark   d1,d3
```
d3 = "spark and hadoop and spark" has 5 words but only **3 distinct** → the mapper emits **3 lines** because of `set()`.

**Marking:** mapper 3 (−1 without set) · reducer 4 (−2 without final flush) · output 1.5 · d3 explanation 0.5.

### C2 (6) — verified
```
['and' 'fast' 'hadoop' 'is' 'reliable' 'spark']
(1, 6)
```
Row = `[0.403, 0.806, 0, 0.307, 0, 0.307]` → non-zero: **and, fast, is, spark**.
**fast** is largest: it appears **twice** in the new text (TF = 2) and it was in only one training doc (high IDF).
**"cheap"** and **"very"** are not in the fitted vocabulary → silently **ignored**. Shape stays (1, 6) because `transform` reuses the old vocabulary.

### C3 (6)
```python
avg_scores = X.mean(axis=0).A1
top_indices = avg_scores.argsort()[-10:][::-1]
top_terms = [feature_names[i] for i in top_indices]
top_values = avg_scores[top_indices]
```
- `argsort()` → indices that sort ascending (smallest first)
- `[-10:]` → the last 10 = the 10 largest
- `[::-1]` → reverse → largest first
- `[::-1]` again in `barh` because barh draws the first item at the **bottom**; reversing puts the top term at the top.

### C4 (7)
```python
# (a)
sparsity = 100.0 * (1.0 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]))

# (b)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_tfidf)
terms = tfidf_vec.get_feature_names_out()
for c in range(4):
    top = kmeans.cluster_centers_[c].argsort()[::-1][:8]
    print(c, [terms[i] for i in top])

# (c)
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    row = matrix[doc_id].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = row.argsort()[::-1][:top_n]
    return [(terms[i], float(row[i])) for i in top_idx if row[i] > 0]
```

### C5 (4)
The vectorizer learns vocabulary and IDF from **all** text, including the test rows → test information leaks into training. Split **raw text first**:
```python
X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_txt)     # fit on train only
X_test = vectorizer.transform(X_test_txt)           # transform test
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```
(Cleaner alternative: `Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])`.)

### C6 (3)
```bash
hdfs dfs -mkdir -p /user/khizar/wc-in
hdfs dfs -put sample.txt /user/khizar/wc-in/
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
  -input /user/khizar/wc-in -output /user/khizar/wc-out \
  -mapper "python3 mapper.py" -reducer "python3 reducer.py" \
  -file mapper.py -file reducer.py
hdfs dfs -cat /user/khizar/wc-out/part-*
```
If `wc-out` already exists, the job **fails** — Hadoop won't overwrite an output directory. Delete it first with `hdfs dfs -rm -r /user/khizar/wc-out`.

---

## SCORING
85+ : ready. 70–84 : fix the gaps, redo missed questions in 2 days. < 70 : revisit `01_CODE_Memorize.md` and `03_NUMERICALS.md`, then redo both papers.
