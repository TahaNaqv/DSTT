# MOCK MID EXAM — PAPER B (High Difficulty)
**Data Science Tools & Techniques · Lectures 9–17 + course repositories**
**Time: 2 hours · Total: 100 marks · Calculator allowed · Attempt ALL questions**
Take this one **2 days after Paper A**, timed and closed-book.

| Section | Type | Marks |
|---|---|---|
| A | MCQs (20 × 1.5) | 30 |
| B | Numericals | 35 |
| C | Coding | 35 |

---

## SECTION A — MCQs (1.5 marks each)

**A1.** Which statement about k-fold cross-validation is correct?
(a) it trains a better model than a single split (b) it gives a more reliable estimate of test performance (c) it removes overfitting (d) each sample is used only once overall

**A2.** A degree-M polynomial fitted with M = 9 to N = 15 points (instead of 10) is described on the slide as:
(a) severe underfit (b) pretty good, slightly overfit (c) identical to M = 3 (d) zero training error

**A3.** Which is a **hyperparameter** of the polynomial curve-fitting algorithm?
(a) w₀ (b) w_M (c) M (d) the training error

**A4.** "Including sales commission as a feature when ranking sales leads" is an example of:
(a) underfitting (b) leakage (c) sampling bias (d) high variance

**A5.** Positive likelihood ratio equals:
(a) TPR / FPR (b) FNR / TNR (c) TP / FP (d) Precision / Recall

**A6.** Specificity is also called:
(a) false acceptance rate (b) positive predictive value (c) true negative rate (d) recall

**A7.** Which imputation method keeps the MNAR signal by adding an extra column?
(a) mean imputation (b) missing indicator (c) listwise deletion (d) kNN

**A8.** ADASYN differs from plain SMOTE because it:
(a) handles categorical features (b) generates more points where the minority class is harder to learn (c) under-samples the majority (d) only works with SVMs

**A9.** For **mild** imbalance (IR below ~10) on tabular data, the decision-guide slide says to try first:
(a) SMOTE-NC (b) EasyEnsemble (c) class weights + tuned threshold (d) anomaly detection

**A10.** `size` vs `count` in a pandas `groupby().agg()`:
(a) identical (b) size counts rows, count ignores NaN (c) count counts rows, size ignores NaN (d) size returns bytes

**A11.** Stratified sampling with **Neyman** allocation gives more samples to strata that are:
(a) smaller (b) larger and more variable (c) less variable (d) chosen by convenience

**A12.** In the Lec 15 comparison, lemmatization (vs stemming) is:
(a) heuristic and fast (b) linguistic and more accurate but slower (c) implemented by PorterStemmer (d) unable to return dictionary words

**A13.** `TfidfVectorizer` treats "helps" and "helpful" as:
(a) the same feature (b) two different features (c) stopwords (d) a bigram

**A14.** `cosine_similarity(query_vec, X)` with 1 query and 10 documents returns shape:
(a) (10, 1) (b) (1, 10) (c) (10, 10) (d) (1,)

**A15.** In the tf-idfexamples save/load script, after loading the vectorizer the new text is processed with:
(a) fit (b) fit_transform (c) transform (d) dump

**A16.** HDFS's NameNode is a single point of failure **unless**:
(a) replication = 3 (b) a Standby NameNode with shared edit logs (HA) is configured (c) block size is 64 MB (d) YARN is installed

**A17.** "Many tiny files" hurt HDFS mainly because:
(a) each file costs NameNode memory (b) DataNodes can't store small blocks (c) replication fails (d) MapReduce can't read them

**A18.** In MapReduce, a failed **worker's** completed map outputs:
(a) are safe in HDFS (b) are lost and those maps re-run (c) are sent to the master (d) are ignored

**A19.** Stream data processing (Lec 17) requires:
(a) multiple passes over the data (b) single-pass/online, small-memory algorithms with irreversible decisions (c) batch storage first (d) relational schemas

**A20.** According to the Lec 17 categories slide, unstructured data is roughly what share of the information being processed?
(a) 5% (b) 10% (c) 50% (d) 80%

---

## SECTION B — NUMERICALS (35 marks)

**B1. TF-IDF and similarity (7 marks)**
D1 = "spark streams data fast", D2 = "hadoop stores data", D3 = "spark beats hadoop", D4 = "data data everywhere".
(a) DF of every term. (2)
(b) Using TF × log₁₀(N/DF), score every term of D1. Which is (are) the most distinctive? (3)
(c) Without computing, state the cosine similarity between D3 and D4 and justify. (1)
(d) In sklearn, what is the IDF of "data"? (1)

**B2. Working backwards (7 marks)**
1,000 patients are screened; 200 truly have the disease. The test has **sensitivity 0.85** and **specificity 0.90**.
(a) Find TP, FN, TN, FP. (3)
(b) Compute precision, accuracy, F1 and NPV. (3)
(c) A doctor says "85% sensitivity, so 85% of positive results are real". Correct them with a number. (1)

**B3. Bias–variance (5 marks)**
True value = 60. Model R (5 datasets): 52, 48, 51, 49, 50. Model S: 30, 90, 55, 75, 50.
Compute Bias² and Variance for both, give the expected error ignoring σ², and diagnose each.

**B4. Outliers (4 marks)**
Data: 22, 25, 27, 28, 30, 31, 33, 35, 36, 100, 2
(a) Q1, Q3, IQR, fence; list outliers. (3)
(b) With μ ≈ 33.55, σ ≈ 22.80, does the z-rule flag 100 or 2? (1)

**B5. Cross-validation & tuning (4 marks)**
(a) 5-fold accuracies: 0.70, 0.74, 0.68, 0.72, 0.76. Report T̂ ± SE (use sample SD). (2)
(b) A grid has 4 values of C, 3 of `k_neighbors`, 2 of `select__k`, with 10-fold CV. How many model fits? (1)
(c) 120 rows, 5-fold: how many rows train each model? (1)

**B6. Sampling (4 marks)**
(a) 99% confidence (z = 2.576), p unknown, margin e = 0.04. Required n₀? With population N = 1500? (2)
(b) n = 230; strata X (N = 5000, σ = 4), Y (N = 3000, σ = 2), Z (N = 2000, σ = 10). Give proportional and Neyman allocations. (2)

**B7. HDFS, SMOTE, MapReduce (4 marks)**
(a) Files of 130 MB, 50 MB and 1 GB; block size 128 MB; replication 2. Total blocks, total replicas, raw disk used. (1.5)
(b) SMOTE in 3-D: x_i = (1.5, 2.0, 6.0), x̂ = (3.5, 1.0, 2.0), λ = 0.6. Find x_new. (1)
(c) Mapper 1 gets "red blue red", mapper 2 gets "blue green red blue". How many pairs are shuffled **without** and **with** a combiner? Give the final reducer output. (1.5)

---

## SECTION C — CODING (35 marks)

**C1. (9 marks)** Write the **inverted index** mapper and reducer for Hadoop Streaming. Input lines look like `d1<TAB>spark is fast`. Output must be `word<TAB>d1,d3` (doc ids sorted, no duplicates). Then give the output for:
```
d1	spark is fast
d2	hadoop is slow
d3	spark and hadoop and spark
```
How many lines does the mapper emit for d3, and why?

**C2. (6 marks)** Predict the output:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
v.fit(["Spark is fast", "Hadoop is reliable", "Spark and Hadoop"])
print(v.get_feature_names_out())
new = v.transform(["Spark is fast and cheap, very fast"])
print(new.shape)
```
Which entries of `new` are non-zero? Which one is largest and why? What happened to "cheap" and "very"?

**C3. (6 marks)** Complete this bar chart of the top-10 terms, then explain `argsort()`, `[-10:]` and `[::-1]`, and why `[::-1]` is used again inside `barh`.
```python
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
feature_names = vectorizer.get_feature_names_out()
avg_scores = ________________________
top_indices = ________________________
top_terms = [________ for i in top_indices]
top_values = ________________________
plt.barh(top_terms[::-1], top_values[::-1], color='skyblue')
plt.show()
```

**C4. (7 marks)** For the Lab 15 pipeline, given `X_tfidf` and `tfidf_vec`:
(a) Compute matrix sparsity in %. (2)
(b) Fit K-Means with 4 clusters and store labels in `df["cluster"]`; print the top 8 words of each centroid. (3)
(c) Write `explain_document(doc_id, matrix, vectorizer, top_n=5)` returning a list of `(term, weight)` tuples. (2)

**C5. (4 marks)** This classifier has a leakage problem. Rewrite it correctly.
```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

**C6. (3 marks)** Write the sequence of shell commands to: create `/user/khizar/wc-in` in HDFS, upload `sample.txt`, run a Streaming job with `mapper.py`/`reducer.py` writing to `/user/khizar/wc-out`, and print the results. What happens if `wc-out` already exists?
