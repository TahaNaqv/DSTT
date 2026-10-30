# MOCK MID EXAM — PAPER A — SOLUTIONS & MARKING
All numerical answers and code outputs verified by running them.

---

## SECTION A — MCQs

| Q | Ans | Why |
|---|---|---|
| A1 | **b** | Already overfitting (tiny train, big test error). More complexity lowers training error further and pushes test error up the right side of the U-curve. |
| A2 | **c** | Lec 11: E[·] = average over different possible training datasets. |
| A3 | **c** | Missingness depends on the missing value itself (wellness) → MNAR. |
| A4 | **b** | x_new = x_i + λ(x̂ − x_i), λ ∈ [0,1] → always on the segment, inside the convex hull. SMOTE **cannot** go outside. |
| A5 | **c** | Classic mistake slide: resampling before CV → optimistic estimate. |
| A6 | **c** | "e-mail" → tokens "e" and "mail"; "e" is 1 character and the default token pattern needs ≥ 2. Verified: `['filter' 'mail' 'spam']`. |
| A7 | **b** | ln((1+N)/(1+N)) + 1 = 0 + 1 = 1. |
| A8 | **b** | The last key never sees a "change", so without the final flush it's never written. |
| A9 | **b** | Duplicates within one doc would emit repeated pairs (the reducer's `set(docs)` would still dedupe, but the mapper does extra work and the shuffle carries extra data). |
| A10 | **b** | Rack awareness: survives node and rack failure. |
| A11 | **c** | Combiner needs associative + commutative ops: sum, count, max. Median/mean-directly/mode are not. |
| A12 | **c** | FsImage = complete state; EditLogs = recent changes relative to latest FsImage. |
| A13 | **b** | Each price depends on the previous ones (temporal dependency). |
| A14 | **b** | Raw counts scale with length; TF-IDF L2-normalizes every row to length 1. |
| A15 | **b** | TruncatedSVD accepts sparse input; PCA needs dense. |
| A16 | **a** | max_df=0.8 (float = proportion) drops very common words; min_df=2 (int = count) drops words in fewer than 2 docs. |
| A17 | **c** | Prevents the same patient appearing in both train and validation (Lec 12 case study). |
| A18 | **c** | Convenience, quota, purposive, snowball are non-probability. |
| A19 | **d** | Lec 16: Veracity = data in doubt. |
| A20 | **b** | "Bottleneck: reduce phase can't start until map phase is finished." |

---

## SECTION B — NUMERICALS

### B1 (8)
**(a) DF table (2)** — N = 3

| Term | D1 | D2 | D3 | DF |
|---|---|---|---|---|
| cloud | 2 | 0 | 0 | 1 |
| data | 1 | 1 | 1 | 3 |
| storage | 1 | 0 | 1 | 2 |
| pipeline | 0 | 1 | 0 | 1 |
| cluster | 0 | 0 | 1 | 1 |

**(b) Version A, log₁₀(N/DF) (3)**
- cloud: TF 2, IDF = log₁₀(3/1) = 0.4771 → **0.9542**
- data: TF 1, IDF = log₁₀(3/3) = 0 → **0**
- storage: TF 1, IDF = log₁₀(3/2) = 0.1761 → **0.1761**

"cloud" is the most descriptive word of D1 (frequent here, absent elsewhere).

**(c) Lec 14 smoothed (2)**
- cloud: 2 × log₁₀(4/2) = 2 × 0.3010 = **0.6021**
- storage: 1 × log₁₀(4/3) = **0.1249**

**(d) (1)** In version A, log(N/DF) = log(1) = 0 for a word in every document. sklearn uses ln((1+N)/(1+DF)) **+ 1**, so IDF = 1 and "data" keeps weight. Verified sklearn D1 row: cloud 0.901, storage 0.343, data 0.266.

### B2 (8)
**(a) (2)** TP = 70; FP = 150 − 70 = **80**; FN = 100 − 70 = **30**; TN = 2000 − 70 − 80 − 30 = **1820**

| | Fraud | Legit |
|---|---|---|
| Pred fraud | 70 | 80 |
| Pred legit | 30 | 1820 |

**(b) (5)**
- Accuracy = (70 + 1820)/2000 = **0.945**
- Precision = 70/150 = **0.467**
- Recall = 70/100 = **0.700**
- Specificity = 1820/1900 = **0.958**
- F1 = 2(0.467)(0.7)/(0.467 + 0.7) = 0.6533/1.1667 = **0.560**

**(c) (1)** Always "legit": accuracy = 1900/2000 = **0.95** — *higher* than our model — but recall = **0**. Accuracy rewards the majority class; on imbalanced data use precision/recall/F1/PR-AUC.

### B3 (6)
**Model P:** mean = 600/5 = **120** → Bias² = (120 − 120)² = **0**. Deviations −20, 20, 10, −30, 20 → squares 400, 400, 100, 900, 400 = 2200 → Var = **440**.
**Model Q:** sum = 108+110+109+107+111 = 545 → mean = **109** → Bias² = (109 − 120)² = (−11)² = **121**. Deviations −1, 1, 0, −2, 2 → squares 1, 1, 0, 4, 4 = 10 → Var = **2**.

Expected error (ignoring σ²): P = 0 + 440 = **440**; Q = 121 + 2 = **123**. Q is better overall despite its bias.

**(b)** P: zero bias, huge variance → **overfitting** → simplify / regularize / more data. Q: big bias, tiny variance → **underfitting** → more complex model, feature creation, non-linear transforms.

(Marking: full credit for correct Bias² and Var with working.)

### B4 (5)
**(a)** Sorted: 3, 5, 7, 8, 12 | 13, 14, 18, 21, 61 (n = 10, even → halves of 5).
Q1 = median of lower half = **7**; Q3 = median of upper half = **18**; IQR = **11**.
Fence = [7 − 16.5, 18 + 16.5] = **[−9.5, 34.5]** → **61 is an outlier**.

**(b)** z = (61 − 16.2)/15.87 = **2.82 < 3** → **not flagged**. The outlier inflates σ (and μ), which shrinks its own z-score — **masking**. Quartiles barely move, so the IQR rule is robust.

### B5 (8)
**(a)** x̂ − x_i = (6, −6); 0.3 × (6, −6) = (1.8, −1.8); x_new = **(5.8, 8.2)**
**(b)** N_new = r·N_maj − N_min. r = 1: 12,000 − 600 = **11,400**. r = 0.25: 3,000 − 600 = **2,400**.
**(c)** 700/128 = 5.47 → **6 blocks** (5 × 128 = 640, last block = **60 MB**). Replicas = 6 × 3 = **18**. Raw disk ≈ 700 × 3 = **2,100 MB** (the last block uses 60 MB, not 128). **6 map tasks** (one split per block).

---

## SECTION C — CODING

### C1 (10)
```python
# mapper.py
#!/usr/bin/env python3
import sys
for line in sys.stdin:
    for word in line.split():
        print(f"{word.lower()}\t1")
```
```python
# reducer.py
#!/usr/bin/env python3
import sys
current, total = None, 0
for line in sys.stdin:
    key, _, value = line.rstrip("\n").partition("\t")
    if key != current:
        if current is not None:
            print(f"{current}\t{total}")
        current, total = key, 0
    total += int(value)
if current is not None:
    print(f"{current}\t{total}")
```
```bash
cat input.txt | python3 mapper.py | sort | python3 reducer.py
```
**`sort`** replaces the framework's shuffle & sort: it brings all identical keys next to each other so the reducer only needs to detect key changes.

**Marking:** mapper 3 · reducer state machine 4 (−2 if the final flush is missing) · command 2 · explanation 1.

### C2 (6) — verified
```
['big' 'cat' 'dog' 'ran' 'saw' 'the']
[[2 1 1 0 1 0]
 [0 0 1 1 0 1]]
(2, 6)
```
- "A"/"a" vanish (1 character, default token pattern needs ≥ 2).
- "big" appears twice in doc 1 → count **2**.
- Columns alphabetical, everything lowercased.

With `stop_words='english'`: **"the" is removed** → `['big' 'cat' 'dog' 'ran' 'saw']`, shape (2, 5).

### C3 (7) — errors and fixes
1. `vectorizer.fit_transform(query)` — **refits** on the query (new vocabulary, columns won't match X) **and** passes a bare string (iterated character by character). Fix: `q = vectorizer.transform([query])`.
2. `cosine_similarity(q, X)` returns shape (1, n); zipping it pairs every text with the whole row. Fix: `.flatten()`.
3. `sorted(..., key=lambda x: x[1])` sorts **ascending** → least similar first. Fix: `reverse=True`.
4. `pd.DataFrame(X, ...)` — X is sparse. Fix: `pd.DataFrame(X.toarray(), ...)`.
5. `plt` is used but never imported (the same bug is in the repo's `cluster_map_viz.py`). Fix: `import matplotlib.pyplot as plt`.

```python
import matplotlib.pyplot as plt
def rank(query):
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, X).flatten()
    return sorted(zip(df['text'], scores), key=lambda x: x[1], reverse=True)
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
```
**Marking:** 1.5 each for any 4 + 1 for clean fixed code.

### C4 (7)
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf(documents, stop_words="english", ngram_range=(1, 1), max_df=1.0, min_df=1):
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range,
                                 max_df=max_df, min_df=min_df)
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix

def to_dataframe(matrix, vectorizer, titles):
    return pd.DataFrame(matrix.toarray(),
                        columns=vectorizer.get_feature_names_out(), index=titles)

def top_terms_per_document(df, n=5):
    result = {}
    for title in df.index:
        result[title] = df.loc[title].sort_values(ascending=False).head(n).to_dict()
    return result

def rank_documents_for_query(query, vectorizer, matrix, titles):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    return sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)
```
**Marking:** build 2 · frame 1.5 · top 1.5 · rank 2.

### C5 (5) — verified with pandas 2.x
(i) only row **0** → `1.0 2.0 3.0`
(ii) rows **0, 1, 3** survive (only the all-NaN row 2 is dropped)
(iii)
```
     0    1    2
0  1.0  2.0  3.0
1  1.0  3.0  4.0
2  1.0  3.0  4.0
3  1.0  4.0  5.0
```
(column means: col 0 = 1.0, col 1 = 3.0, col 2 = 4.0)
(iv) prints **`None`** (inplace returns nothing; `data` itself now has 0s)
(v) removes **column 4** (the only all-NaN column)

---

## SCORING
- 85+ : exam-ready. 70–84 : redo the questions you missed in 2 days. < 70 : go back to the matching block in `01_CODE_Memorize.md` / topic in `03_NUMERICALS.md`, then try Paper B.
