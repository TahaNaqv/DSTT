# MOCK MID EXAM — PAPER A (High Difficulty)
**Data Science Tools & Techniques · Lectures 9–17 + course repositories**
**Time: 2 hours · Total: 100 marks · Calculator allowed · Attempt ALL questions**
Do not open `05_MOCK_EXAM_A_SOLUTION.md` until you've finished.

| Section | Type | Marks |
|---|---|---|
| A | MCQs (20 × 1.5) | 30 |
| B | Numericals | 35 |
| C | Coding | 35 |

---

## SECTION A — MCQs (1.5 marks each, no negative marking)

**A1.** A model gets 0.4% training error and 18% test error. Increasing the polynomial degree further will most likely:
(a) reduce both errors (b) reduce training error, increase test error (c) increase training error, reduce test error (d) leave test error unchanged

**A2.** In `E[(Y − f̂(X))²] = Bias² + Var + σ²`, the expectation is taken over:
(a) features (b) test points only (c) different possible training datasets (d) classes

**A3.** Patients who feel unwell are more likely to skip a follow-up survey, and *wellness* is exactly the variable that's missing. This is:
(a) MCAR (b) MAR (c) MNAR (d) systematic sampling

**A4.** Which statement about SMOTE is FALSE?
(a) new points lie on segments between minority neighbours (b) it can create points outside the minority region (c) features should be scaled first (d) default k is 5

**A5.** A team runs SMOTE on the whole dataset, then 5-fold CV, and reports F1 = 0.94. The true F1 on fresh data is most likely:
(a) about the same (b) higher (c) lower, because validation folds contained synthetic neighbours of training rows (d) undefined

**A6.** `TfidfVectorizer()` (default settings) is fitted on `["e-mail spam filter", "spam e-mail"]`. The vocabulary is:
(a) ['e-mail','filter','spam'] (b) ['e','filter','mail','spam'] (c) ['filter','mail','spam'] (d) ['email','filter','spam']

**A7.** In sklearn, a word appearing in **every** document has IDF:
(a) 0 (b) 1 (c) ln(N) (d) undefined

**A8.** In the word-count reducer, the line after the `for` loop (`if current is not None: out.write(...)`) exists because:
(a) the first word would otherwise be duplicated (b) the last word never triggers a key change, so it would be lost (c) Hadoop requires a footer (d) it handles empty lines

**A9.** The inverted-index mapper uses `set(text.lower().split())`. Removing `set()` would:
(a) crash the reducer (b) emit a word-doc pair once per occurrence instead of once per document (c) remove stop words (d) sort the output

**A10.** HDFS places the 3 replicas of a block as:
(a) 3 nodes on 3 racks (b) writer's node; another rack; same rack as replica 2 on a different node (c) all on the writer's rack (d) random nodes

**A11.** Which operation can safely be used as a MapReduce **combiner**?
(a) median (b) mean computed directly (c) maximum (d) mode of strings

**A12.** The NameNode's **EditLogs** contain:
(a) block data (b) the complete namespace since start-up (c) recent modifications relative to the latest FsImage (d) DataNode heartbeats

**A13.** Stock prices over consecutive days are non-IID mainly because they are:
(a) not identically distributed only (b) not independent (c) uniformly distributed (d) categorical

**A14.** Why does the Lab 15 README say K-Means fails on raw Bag-of-Words for variable-length posts?
(a) BoW is dense (b) long posts have much larger vector norms and dominate Euclidean distances (c) BoW removes stopwords (d) K-Means needs labels

**A15.** You need a 2-D map of a 4000-column **sparse** TF-IDF matrix without converting it to dense. Use:
(a) PCA (b) TruncatedSVD (c) t-SNE on `.toarray()` (d) StandardScaler

**A16.** `TfidfVectorizer(max_df=0.8, min_df=2)` removes words that appear in:
(a) more than 80% of docs OR fewer than 2 docs (b) fewer than 80% of docs (c) exactly 2 docs (d) more than 2 docs

**A17.** A hospital dataset has up to 12 admissions per patient. The correct CV splitter is:
(a) KFold(shuffle=True) (b) StratifiedKFold (c) GroupKFold on patient_id (d) LeaveOneOut

**A18.** Which is a **non-probability** sampling design?
(a) systematic (b) cluster (c) quota (d) multi-stage

**A19.** "Data in doubt" — uncertainty from inconsistency, incompleteness, ambiguity — is which V?
(a) Volume (b) Velocity (c) Variety (d) Veracity

**A20.** In MapReduce, the main bottleneck noted in Lec 16 is:
(a) mappers can't run in parallel (b) the reduce phase can't start until the map phase finishes (c) HDFS can't store large files (d) keys must be integers

---

## SECTION B — NUMERICALS (35 marks)

**B1. TF-IDF (8 marks)**
D1 = "cloud data cloud storage", D2 = "data pipeline", D3 = "storage cluster data".
(a) Build the DF table for all terms. (2)
(b) Using **TF-IDF = TF × log₁₀(N/DF)**, compute the score of every term in D1. (3)
(c) Using the **Lec 14 smoothed** formula IDF = log₁₀[(1+N)/(1+DF)], compute TF-IDF of "cloud" and "storage" in D1. (2)
(d) Why does "data" get exactly 0 in (b), but sklearn gives it a non-zero weight? (1)

**B2. Confusion matrix (8 marks)**
2,000 transactions, 100 are fraud. The model flags 150 as fraud; 70 of the flagged are really fraud.
(a) Construct the confusion matrix. (2)
(b) Compute accuracy, precision, recall, specificity and F1. (5)
(c) A lazy model that always says "legitimate" — what is its accuracy and recall? What does this show? (1)

**B3. Bias and variance (6 marks)**
True value at a test point = 120. Two models are each trained on 5 different datasets.
- Model P predicts: 100, 140, 130, 90, 140
- Model Q predicts: 108, 110, 109, 107, 111
(a) Compute Bias² and Variance for both. (4)
(b) Diagnose each (under/overfit) and give one fix for each. (2)

**B4. Outliers (5 marks)**
Data: 3, 7, 8, 5, 12, 14, 21, 13, 18, 61
(a) Find Q1, Q3, IQR and the IQR fence. Which values are outliers? (3)
(b) Given μ = 16.2 and σ = 15.87, is 61 flagged by the |z| > 3 rule? Explain the discrepancy. (2)

**B5. SMOTE & HDFS (8 marks)**
(a) x_i = (4, 10), chosen neighbour x̂ = (10, 4), λ = 0.3. Find x_new. (2)
(b) Training set: 12,000 majority, 600 minority. How many synthetic points does SMOTE create for target ratio r = 1? For r = 0.25? (2)
(c) A 700 MB file is stored in HDFS (128 MB blocks, replication 3). How many blocks? Size of the last block? Total replicas? Raw disk used? How many map tasks? (4)

---

## SECTION C — CODING (35 marks)

**C1. (10 marks)** Write a Hadoop-Streaming **mapper.py** and **reducer.py** in Python that count word frequencies (case-insensitive). Then write the one-line command to test them locally on `input.txt`, and explain what replaces the shuffle-and-sort step.

**C2. (6 marks)** Predict the exact output:
```python
from sklearn.feature_extraction.text import CountVectorizer
docs = ["A big cat saw a big dog", "The dog ran"]
vec = CountVectorizer()
X = vec.fit_transform(docs)
print(vec.get_feature_names_out())
print(X.toarray())
print(X.shape)
```
Then: how does the vocabulary change with `CountVectorizer(stop_words='english')`?

**C3. (7 marks)** The following code has **at least 4 errors** (bugs or bad practice). Find and fix each.
```python
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('data/sample_docs.csv')
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

def rank(query):
    q = vectorizer.fit_transform(query)
    scores = cosine_similarity(q, X)
    return sorted(zip(df['text'], scores), key=lambda x: x[1])

tfidf_df = pd.DataFrame(X, columns=vectorizer.get_feature_names_out())
sns.clustermap(tfidf_df, cmap='coolwarm')
plt.show()
```

**C4. (7 marks)** Write the four helper functions of the class-activity repo — `build_tfidf`, `to_dataframe`, `top_terms_per_document`, `rank_documents_for_query` — with correct parameters and return values.

**C5. (5 marks)** Given:
```python
import pandas as pd
from numpy import nan as NaN
data = pd.DataFrame([[1, 2, 3], [1, NaN, NaN], [NaN, NaN, NaN], [NaN, 4, 5]])
```
Write the output (or the surviving row indices) of:
(i) `data.dropna()` (ii) `data.dropna(how='all')` (iii) `data.fillna(data.mean(skipna=True))` (iv) `print(data.fillna(0, inplace=True))` (v) After `data[4] = NaN`, what does `data.dropna(axis=1, how='all')` remove?
