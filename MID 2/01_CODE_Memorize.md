# 01 — CODE: Everything to Memorize (in the order to learn it)

**Covers:** all 5 of sir's repos + every code snippet on the Lec 9–17 slides.
Every output shown below was produced by actually running the code — not guessed.

---

## HOW TO MEMORIZE THIS IF YOU'RE "BAD AT MEMORIZING"

You don't need to memorize 30 programs. You need to memorize **3 skeletons**. Everything else is a skeleton plus 1–3 extra lines.

| Skeleton | Mnemonic | Used in |
|---|---|---|
| **A. TF-IDF skeleton** (4 lines) | **I-C-F-I** → "**I C**an **F**ind **I**t" = **I**mport, **C**reate, **F**it_transform, **I**nspect | 4 of the 5 repos |
| **B. Mapper** (3 lines) | "**For line, for word, print word-tab-1**" | MapReduce repo |
| **C. Reducer** (state machine) | "**Split → Changed? → Flush → Reset → Add → Last flush**" | MapReduce repo |

**The method (do it in this order, don't skip):**
1. **Read** a block below and understand *why* each line exists (the "why" is what sticks).
2. **Cover the code**, write it on paper from memory.
3. **Check** against this file. Circle what you missed. Rewrite only the missed lines 3×.
4. Next day: do the matching drill in `02_CODE_Drills.md` (fill-in-the-blank → blank page).
5. Say the mnemonic out loud before writing — your hand follows your mouth.

**The golden learning order** (each block reuses the previous one, so you never learn twice):

```
BLOCK 1  Pandas basics (Lec 13)                 ← foundation, everything uses df
BLOCK 2  Text cleaning: tokenize/stopwords/stem/lemma (Lec 15)
BLOCK 3  Bag of Words — CountVectorizer (Lec 15)
BLOCK 4  ★ TF-IDF SKELETON (basic_tfidf.py)      ← learn this one cold
BLOCK 5  TF-IDF + 1 extra idea (tf-idfexamples repo: similarity, top terms, classifier, save/load)
BLOCK 6  Class activity (unstructuredata_tfidf_repo: 4 helper functions)
BLOCK 7  Visualization (tfidf-visualization repo: bar, heatmap, wordcloud, PCA, clustermap, comparative)
BLOCK 8  Full pipeline (textanalysis_tfidf Lab 15: sparsity, KMeans, TruncatedSVD, explain_document)
BLOCK 9  ★ MapReduce: mapper + reducer + inverted index + Hadoop commands
BLOCK 10 Lec 12 slide code (imputers, outliers, SMOTE, pipelines) — lower priority
```

**Priority:** Blocks 4, 5, 6, 9 are the highest-yield because sir named those repos. Block 10 is slide code, not a repo — learn it last.

---

# BLOCK 1 — Pandas basics (Lec 13)

Everything else creates or reads a DataFrame, so this comes first.

### 1.1 Series and DataFrame
```python
from pandas import Series, DataFrame
import pandas as pd

obj = Series([4, 7, -5, 3])
print(obj)          # index 0..3 with values
print(obj.index)    # RangeIndex(start=0, stop=4, step=1)
print(obj.values)   # [ 4  7 -5  3]

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year':  [2000, 2001, 2002, 2001, 2002],
        'pop':   [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)      # dict keys become columns
```
- **Series** = 1-D, values + index. Default index is **0 to N−1**.
- **DataFrame** = 2-D table; each column can have a different dtype; has row AND column index.

### 1.2 Inspect, select, filter
```python
df = pd.read_csv("adult.csv")
df.head(); df.tail()    # first/last rows
df.shape                # (rows, columns)
df.columns              # column names
df.describe()           # summary statistics
df.info()

df['age']               # one column  -> Series
df[['age', 'income']]   # two columns -> DataFrame (double brackets!)
df.iloc[0:5]            # rows by POSITION (0,1,2,3,4 — end excluded)
df[df['age'] > 50]      # boolean filter
```

### 1.3 Missing data
```python
df.isnull().sum()            # count NaN per column
df['income'].fillna('Unknown')
df.dropna()                  # drop rows with ANY NaN

from numpy import nan as NaN
data = Series([1, NaN, 2.5, NaN, 6])
data.notnull()               # True/False mask
data[data.notnull()]         # same result as data.dropna()

data = DataFrame([[1, 2, 3], [1, NaN, NaN], [NaN, NaN, NaN], [NaN, 4, 5]])
data.dropna()                    # keeps ONLY row 0 (the only complete row)
data.dropna(how='all')           # drops ONLY row 2 (the all-NaN row)
data[4] = NaN                    # add an all-NaN column named 4
data.dropna(axis=1, how='all')   # drops column 4
data.fillna(0)                   # NaN -> 0
data.fillna(data.mean(skipna=True))   # NaN -> that column's mean
data.fillna(0, inplace=True)     # modifies data itself, returns None
```
**Memory hook for `dropna`:** default = "any NaN kills the row". `how='all'` = "only an all-NaN row dies". `axis=1` = "do it to columns instead".

**Trap:** `print(data.fillna(0, inplace=True))` prints **`None`** in pandas 2.x (the version the course uses) — inplace modifies `data` and returns nothing. (Pandas 3.0 changed this to return the DataFrame; answer `None` in the exam unless told otherwise.) Verified: `data.dropna()` keeps only row 0; `data.dropna(how='all')` keeps rows 0, 1, 3; `fillna(data.mean())` fills column 1 with 3.0 and column 2 with 4.0.

### 1.4 Transform
```python
df.rename(columns={'education-num': 'edu_num'}, inplace=True)
df['income_flag'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
pd.get_dummies(df['gender'])      # one-hot encoding
df['age'].hist()                  # histogram
df['income'].value_counts().plot(kind='bar')
```

### 1.5 Home Activities (Lec 13) — likely "write the code" question
```python
import numpy as np, pandas as pd, math

arr = np.random.randint(1, 100, (3, 5))                  # 3 rows x 5 cols, values 1..99
df = pd.DataFrame(arr, index=['a', 'b', 'c'], columns=[1, 2, 3, 4, 5])
df = df.T                                                # transpose -> 5 x 3
df[df < 40] = 0                                          # values < 40 become 0
df_sqrt = df.apply(lambda col: col.apply(math.sqrt))     # Activity 2: sqrt of every value
# simpler equivalent: df.applymap(math.sqrt)   or   np.sqrt(df)
```
**Trap:** `np.random.randint(1, 100)` — the high value **100 is excluded**, so values are 1–99.

### 1.6 SQL into pandas (Lec 14)
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("enterprise_sales.db")
query = """
SELECT region, COUNT(*) AS tx_count, AVG(revenue) AS avg_rev
FROM transactions
WHERE tx_date >= '2026-01-01'
GROUP BY region
"""
df = pd.read_sql_query(query, conn)
print(df.head())
```
**Why:** structured data = fixed schema + ACID + fast indexed aggregation → lands directly in a DataFrame ready for ML.

---

# BLOCK 2 — Text cleaning (Lec 15 + Lab 15)

**NLP pipeline order (memorize):** Text Cleaning → Tokenization → Stopword Removal → Stemming/Lemmatization → Feature Extraction → Modeling

### 2.1 Stemming
```python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words = ["running", "runs", "easily", "fairly"]
print([stemmer.stem(w) for w in words])
# ['run', 'run', 'easili', 'fairli']      <- verified output
```
Stemming chops suffixes by rules, so it can produce **non-words** (`easili`, `fairli`, `studi`).

### 2.2 Lemmatization
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
words = ["running", "ran", "better"]
for w in words:
    print(w, '→', lemmatizer.lemmatize(w, pos='v'))
# running → run
# ran → run
# better → better      (as a VERB 'better' stays 'better'; you'd need pos='a' to get 'good')
```
Lemmatization returns a **real dictionary word**. `pos='v'` tells it "treat as verb" — that's why `ran → run` works.

| | Stemming | Lemmatization |
|---|---|---|
| Approach | Heuristic (rules) | Linguistic (dictionary) |
| Accuracy | Low | High |
| Speed | Fast | Slower |
| Library | NLTK Porter, Snowball | WordNet |
| `studies` | `studi` | `study` |

### 2.3 The Lab 15 cleaning step (solution to TODO 1.1–1.3)
```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# TODO 1.1 tokenize, lowercase, keep alphabetic only
tokens = [w.lower() for w in word_tokenize(sample_post) if w.isalpha()]

# TODO 1.2 remove stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3 stem and lemmatize
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.lemmatize(w, pos='v') for w in filtered_tokens]
```
**Memory hook:** it's **three list comprehensions**, each feeding the next: *tokens → filtered → stemmed/lemmatized*. The pattern is always `[DO(w) for w in PREVIOUS if CONDITION]`.

**Why `set(...)` for stopwords:** checking `w in a_set` is instant; checking a list is slow.

Slide example: `"Text analysis is fun and informative!"` → tokens `['Text','analysis','is','fun','and','informative']` → after stopwords `['Text','analysis','fun','informative']`.

---

# BLOCK 3 — Bag of Words (Lec 15)

```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = ["I love data science", "Data science is amazing"]
vec = CountVectorizer()
X = vec.fit_transform(corpus)
print(vec.get_feature_names_out())
print(X.toarray())
```
**Verified output:**
```
['amazing' 'data' 'is' 'love' 'science']
[[0 1 0 1 1]
 [1 1 1 0 1]]
```
Notice three things the examiner loves:
1. **"I" disappeared.** Default `token_pattern` keeps only tokens of **2+ characters**, so "I" and "a" are dropped.
2. **Lowercased automatically** — "Data" and "data" are the same column.
3. **Columns are alphabetical.**

BoW: `X[i][j]` = count of word j in document i. Ignores grammar and order; produces sparse, high-dimensional matrices.

---

# BLOCK 4 — ★ THE TF-IDF SKELETON (learn this cold)

**File:** `tf-idfexamples/examples/basic_tfidf.py`. **Mnemonic: I-C-F-I.**

```python
from sklearn.feature_extraction.text import TfidfVectorizer      # I — Import

documents = [
    "Data science is fun",
    "Python makes data analysis easy",
    "TF-IDF is a feature extraction method"
]

vectorizer = TfidfVectorizer()                  # C — Create
X = vectorizer.fit_transform(documents)         # F — Fit_transform

print("Feature Names:", vectorizer.get_feature_names_out())   # I — Inspect
print("\nTF-IDF Matrix:\n", X.toarray())
```
**Verified:** 13 features, `X.shape = (3, 13)`:
`['analysis' 'data' 'easy' 'extraction' 'feature' 'fun' 'idf' 'is' 'makes' 'method' 'python' 'science' 'tf']`
- `"TF-IDF"` becomes **two** tokens, `tf` and `idf` (the hyphen splits it).
- `"a"` is dropped (1 character).

### What each line actually does
| Line | Meaning |
|---|---|
| `TfidfVectorizer()` | Creates an empty, unfitted object. No vocabulary yet. |
| `.fit_transform(docs)` | **fit** = learn vocabulary + IDF of each word. **transform** = turn docs into the matrix. |
| `X` | A **sparse** matrix (stores only non-zeros). Shape = (n_docs, n_unique_terms). |
| `.get_feature_names_out()` | The vocabulary = column names, alphabetical. |
| `.toarray()` | Sparse → dense NumPy array so you can print/see it. |

### The 4 rules that come up in every TF-IDF question
1. **`fit_transform` on training docs; only `transform` on new docs/queries.** If you re-fit on a query, you get a brand-new vocabulary and the columns no longer line up.
2. **sklearn's IDF is smoothed and uses natural log:** `idf(t) = ln((1+N)/(1+df)) + 1`
3. **Each row is L2-normalized** (row length = 1). That's why scores look like 0.5, 0.7 not 1.4.
4. A word in **every** document gets the **lowest** IDF (exactly 1.0 in sklearn, 0 in the slide formula).

### Same skeleton on the Lec 15 corpus (verified)
```python
corpus = ["I love data science", "Data science is amazing"]
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)
# ['amazing' 'data' 'is' 'love' 'science']
# [[0.    0.502 0.    0.705 0.502]
#  [0.576 0.41  0.576 0.    0.41 ]]
# tfidf.idf_ = [1.405, 1.0, 1.405, 1.405, 1.0]
```
`data` and `science` are in both docs → idf = 1.0 → lower weight than the unique words.

### Variant with stopwords (Lec 15 toolkit slide)
```python
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(corpus)
print(tfidf.get_feature_names_out())
# ['amazing' 'data' 'love' 'science']      <- 'is' removed (stopword)
```
(The slide shows `['amazing', 'data', 'science']`, which also drops "love" — the real verified output keeps "love". If asked, write what the code actually produces: 4 terms.)

### Lec 14 example (same skeleton)
```python
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
docs = ["This text is helpful", "It helps understand the data science"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())
print(X.toarray())
# 10 terms; no word is shared, so doc1 has four 0.5 values, doc2 has six 0.408 values
```
Note `helpful` and `helps` are **different columns** — TfidfVectorizer does **not** stem.

---

# BLOCK 5 — tf-idfexamples repo: skeleton + ONE new idea each

Learn only the **new lines** (marked `# NEW`). The rest is Block 4.

### 5.1 Document similarity — `tfidf_similarity.py`
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity          # NEW

corpus = [
    "Machine learning enables computers to learn",
    "Deep learning is part of machine learning",
    "Natural language processing is fun"
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

sim_matrix = cosine_similarity(X, X)                             # NEW
print("\nCosine Similarity Matrix:\n", sim_matrix)
```
**Verified output:**
```
[[1.    0.3   0.   ]
 [0.3   1.    0.106]
 [0.    0.106 1.   ]]
```
- Diagonal = **1** (every doc identical to itself). Matrix is **symmetric**.
- Doc 0 vs Doc 2 = **0** → they share no words.
- Doc 1 vs Doc 2 = 0.106 → they share only "is".
- Shape is (3, 3) = n_docs × n_docs.

### 5.2 Top terms / feature selection — `tfidf_feature_selection.py`
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [
    "Artificial intelligence and machine learning",
    "Deep learning with neural networks",
    "AI applications in healthcare"
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

avg_tfidf = X.mean(axis=0).A1                                    # NEW
terms = vectorizer.get_feature_names_out()
ranking = sorted(zip(terms, avg_tfidf), key=lambda x: x[1], reverse=True)   # NEW

print("Top TF-IDF Terms:")
for term, score in ranking[:10]:
    print(f"{term:20s}: {score:.4f}")
```
**Verified:** top term is **`learning` (0.2370)** — the only word that appears twice.

Decode the two new lines:
- `X.mean(axis=0)` → average of each **column** (each word) across all docs. `axis=0` = "go down the rows".
- `.A1` → flattens the 1×n matrix into a plain 1-D array.
- `zip(terms, avg_tfidf)` → pairs `(word, score)`.
- `sorted(..., key=lambda x: x[1], reverse=True)` → sort by score (`x[1]`), highest first.

**Spot-the-bug alert:** in the actual repo file the `print` inside the `for` loop has **lost its indentation** → `IndentationError`. A classic "find the error" question.

### 5.3 Sentiment classifier — `tfidf_sentiment_classifier.py`
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split            # NEW
from sklearn.linear_model import LogisticRegression             # NEW
from sklearn.metrics import classification_report               # NEW

data = pd.DataFrame({
    'text': ['I love this movie', 'This film was terrible', 'Amazing acting and story',
             'Worst movie ever', 'Highly recommend this film'],
    'label': [1, 0, 1, 0, 1]
})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(classification_report(y_test, preds))
```
**Mnemonic: "Split → Fit → Predict → Report".**
- `test_size=0.2` on 5 rows → **1 test row, 4 train rows**. Far too small to trust the report.
- **Hidden leak (exam-worthy):** the vectorizer is fitted on **all** text **before** the split, so test-set words influence the IDF. Lec 12's rule: *split first, fit transformers on train only*. The correct order is: split the raw text, then `fit_transform(X_train_text)`, then `transform(X_test_text)`.
- `classification_report` prints precision, recall, F1, support per class — Lec 11 metrics.

### 5.4 Save / load — `tfidf_save_load.py`
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump, load                                    # NEW

texts = ["TF-IDF saves models", "Reuse vectorizers efficiently"]
vec = TfidfVectorizer()
X = vec.fit_transform(texts)

dump(vec, 'vectorizer.joblib')                                   # NEW save
vec_loaded = load('vectorizer.joblib')                           # NEW load
X2 = vec_loaded.transform(["Save and load TF-IDF models"])       # transform ONLY
print("Transformed vector shape:", X2.shape)
```
**Verified:** vocabulary = `['efficiently','idf','models','reuse','saves','tf','vectorizers']` (7) → **`X2.shape = (1, 7)`**.
- 1 row because 1 new document; 7 columns because the **old** vocabulary is reused.
- "Save", "and", "load" are not in the vocabulary → ignored. "save" ≠ "saves" (no stemming).
- Only `tf`, `idf`, `models` match → row = `[0, 0.577, 0.577, 0, 0, 0.577, 0]`.

**Why save:** so you can transform future text with the *identical* vocabulary and IDF weights.

---

# BLOCK 6 — unstructuredata_tfidf_repo (class activity) — 4 functions

This is the Lec 14 "Class Task". The four functions in `src/tfidf_utils.py` are **the** functions to know. **Mnemonic: "Build → Frame → Top → Rank".**

```python
from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1) BUILD
def build_tfidf(documents, stop_words="english", ngram_range=(1, 1), max_df=1.0, min_df=1):
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


# 2) FRAME — sparse matrix -> labeled DataFrame
def to_dataframe(matrix, vectorizer, titles):
    return pd.DataFrame(
        matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=titles,
    )


# 3) TOP — top-n terms per document
def top_terms_per_document(df, n=5):
    result = {}
    for title in df.index:
        top = df.loc[title].sort_values(ascending=False).head(n)
        result[title] = top.to_dict()
    return result


# 4) RANK — cosine similarity of a query against every document
def rank_documents_for_query(query, vectorizer, matrix, titles):
    query_vec = vectorizer.transform([query])                     # transform, NOT fit!
    scores = cosine_similarity(query_vec, matrix).flatten()
    return sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)
```

### Line-by-line "why" for the hard ones
- `vectorizer.transform([query])` — **list** with one string, because the vectorizer expects a collection of documents. Using `fit_transform` here would be **wrong**: it builds a new vocabulary from the query alone.
- `cosine_similarity(query_vec, matrix)` → shape **(1, 10)** (1 query × 10 docs). `.flatten()` → 1-D array of 10 scores.
- `df.loc[title]` → that document's row (one score per word). `.sort_values(ascending=False).head(n)` → its n highest-weighted words = an automatic summary.

### Parameters you must be able to explain
| Parameter | Meaning |
|---|---|
| `stop_words="english"` | Remove common English words (the, and, is…) |
| `ngram_range=(1, 2)` | Use single words **and** two-word phrases ("climate change") |
| `max_df=0.8` | Ignore words in **more than 80%** of docs (too common to be useful) |
| `min_df=2` | Ignore words in **fewer than 2** docs (too rare / typos) |

`max_df` as a **float** = proportion of documents; `min_df` as an **int** = absolute count of documents.

### Verified results on the 10-document dataset
- Default (`stop_words="english"`): **shape (10, 537)**
- With `ngram_range=(1,2), max_df=0.8, min_df=2`: **shape (10, 69)** — `min_df=2` throws away every word used in only one document, so the vocabulary collapses.
- Query `"renewable energy and climate policy"` → **Climate Change (0.323)** first. Query `"password security and phishing"` → **Cybersecurity (0.176)**. Query `"healthy sleep exercise and diet"` → **Public Health (0.285)**.
- Most documents score **0.000** for a query — they share none of its words.

### The runner (`run_activity.py`) — just calls the 4 functions in order
```python
titles, documents = load_documents()
vectorizer, matrix = build_tfidf(documents)
print("TF-IDF matrix shape:", matrix.shape)
df = to_dataframe(matrix, vectorizer, titles)
print(df.iloc[:3, :10].round(3))                      # first 3 docs x 10 terms
for title, terms in top_terms_per_document(df, n=5).items():
    print(title, terms)
for q in ["renewable energy and climate policy", "password security and phishing"]:
    for title, score in rank_documents_for_query(q, vectorizer, matrix, titles):
        print(f"  {title:25s} {score:.3f}")
```

### The test file — what "correct" means
```python
assert len(titles) == 10 and len(docs) == 10
assert all(len(d.split()) > 30 for d in docs)                   # "long text"
assert matrix.shape[0] == 10
assert matrix.shape[1] == len(vectorizer.get_feature_names_out())
assert ranking[0][0] == "Climate Change"                         # top result's title
```
`ranking[0][0]` = first tuple, first element = the title.

---

# BLOCK 7 — tfidf-visualization repo (6 scripts)

Every script starts with the **same 4 lines**. Memorize them once:
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('data/sample_docs.csv')                 # columns: text, category
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
```
Then each script adds a different "plot ending". **Mnemonic: "Bar, Heat, Cloud, PCA, Cluster, Compare".**

| Script | Shows | The key new line(s) |
|---|---|---|
| bar_chart | Top 10 terms | `avg_scores = X.mean(axis=0).A1` + `argsort()[-10:][::-1]` + `plt.barh` |
| heatmap | Docs × terms weights | `sns.heatmap(tfidf_df, cmap='YlGnBu')` |
| wordcloud | Word size ∝ score | `WordCloud(...).generate_from_frequencies(word_freq)` |
| pca | Docs in 2-D | `PCA(n_components=2).fit_transform(X.toarray())` |
| cluster_map | Clustered heatmap | `sns.clustermap(tfidf_df, cmap='coolwarm')` |
| comparative | Tech vs Business words | `(tech_avg - bus_avg).sort_values(ascending=False)[:10]` |

### 7.1 Bar chart (most likely to be asked)
```python
import matplotlib.pyplot as plt
feature_names = vectorizer.get_feature_names_out()
avg_scores = X.mean(axis=0).A1                         # average score of each word
top_indices = avg_scores.argsort()[-10:][::-1]         # indices of 10 highest, descending
top_terms = [feature_names[i] for i in top_indices]
top_values = avg_scores[top_indices]

plt.figure(figsize=(8, 5))
plt.barh(top_terms[::-1], top_values[::-1], color='skyblue')
plt.xlabel('TF-IDF Score')
plt.title('Top 10 TF-IDF Terms')
plt.tight_layout()
plt.show()
```
**Decode `argsort()[-10:][::-1]`** (this is asked a lot):
- `argsort()` → indices that would sort the array **ascending** (smallest first).
- `[-10:]` → last 10 = the 10 **largest**.
- `[::-1]` → reverse → largest first.

Why reverse **again** inside `barh` (`[::-1]`)? `barh` draws the first item at the **bottom**; reversing puts the biggest bar on **top**.

### 7.2 Heatmap
```python
import seaborn as sns
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
plt.figure(figsize=(10, 6))
sns.heatmap(tfidf_df, cmap='YlGnBu')
plt.title('TF-IDF Heatmap (Documents × Terms)')
plt.xlabel('Terms'); plt.ylabel('Documents')
plt.show()
```

### 7.3 Word cloud
```python
from wordcloud import WordCloud
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = X.toarray().sum(axis=0)                  # total score per word
word_freq = dict(zip(feature_names, tfidf_scores))      # {word: score}
wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('TF-IDF WordCloud')
plt.show()
```
Note: bar chart uses **mean**, word cloud uses **sum** (`axis=0` in both = per word).

### 7.4 PCA scatter
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(X.toarray())                # PCA needs DENSE input
plt.scatter(reduced[:, 0], reduced[:, 1], c=pd.factorize(df['category'])[0], cmap='viridis')
for i, txt in enumerate(df['category']):
    plt.annotate(txt, (reduced[i, 0], reduced[i, 1]))
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.show()
```
- `.toarray()` because PCA wants a dense matrix (TruncatedSVD works on sparse — see Block 8).
- `pd.factorize(...)[0]` turns category names into integer codes (Technology=0, Business=1) for colours.

### 7.5 Cluster map
```python
import seaborn as sns
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
sns.clustermap(tfidf_df, cmap='coolwarm', figsize=(10, 8))
plt.title('TF-IDF Cluster Map')
plt.show()
```
**Spot-the-bug:** the repo file uses `plt` but **never imports matplotlib.pyplot** → `NameError: name 'plt' is not defined`. Fix: add `import matplotlib.pyplot as plt`.

### 7.6 Comparative (Tech vs Business)
```python
tech_docs = df[df['category'] == 'Technology']['text']
business_docs = df[df['category'] == 'Business']['text']
X = vectorizer.fit_transform(pd.concat([tech_docs, business_docs]))
tfidf_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
tech_avg = tfidf_matrix.iloc[:len(tech_docs)].mean()        # first rows = tech
bus_avg = tfidf_matrix.iloc[len(tech_docs):].mean()         # remaining rows = business
diff = (tech_avg - bus_avg).sort_values(ascending=False)[:10]
diff.plot(kind='bar', color='coral')
plt.show()
```
Logic: tech docs stacked first, business after → `iloc` slices them back apart → the difference shows words that are "more tech than business".

---

# BLOCK 8 — textanalysis_tfidf (Lab 15 full pipeline)

This is the biggest script. Learn it as **4 steps**, each one short. **Mnemonic: "Clean → Vectorize → Cluster → Explain".** Step 1 is Block 2.

### Setup (loading the data)
```python
from sklearn.datasets import fetch_20newsgroups
CATEGORIES = ["sci.electronics", "rec.autos", "comp.sys.mac.hardware", "sci.med"]
raw_data = fetch_20newsgroups(subset="all", categories=CATEGORIES,
                              remove=("headers", "footers", "quotes"))
df = pd.DataFrame({"text": raw_data.data,
                   "true_label": [raw_data.target_names[i] for i in raw_data.target]})
df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)   # keep posts > 50 chars
```

### Step 2 — BoW vs TF-IDF + sparsity + top-10 plot
```python
bow_vec   = CountVectorizer(stop_words='english', max_features=4000, min_df=5)
tfidf_vec = TfidfVectorizer(stop_words='english', max_features=4000, min_df=5)
X_bow   = bow_vec.fit_transform(df['text'])
X_tfidf = tfidf_vec.fit_transform(df['text'])

total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))      # % of zeros
print(f"TF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
top_idx = tfidf_means.argsort()[-10:]
top_words = tfidf_vec.get_feature_names_out()[top_idx]
top_scores = tfidf_means[top_idx]
plt.barh(top_words, top_scores, color="teal")
plt.savefig("top_features.png", dpi=150)
```
- `max_features=4000` → keep only the 4000 most frequent words.
- **`X.nnz`** = number of **non-zero** entries. So `nnz / total` = fraction non-zero, and `1 − that` = fraction zero = **sparsity**.
- `np.asarray(...).ravel()` does the same job as `.A1` in Block 5 — two ways to flatten.

### Step 3 — KMeans + centroid words + TruncatedSVD map
```python
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_tfidf)                     # one label per post

terms = tfidf_vec.get_feature_names_out()
for c in range(4):
    top = kmeans.cluster_centers_[c].argsort()[::-1][:8]      # 8 heaviest words of centroid c
    print(c, [terms[i] for i in top])

svd = TruncatedSVD(n_components=2, random_state=42)
coords = svd.fit_transform(X_tfidf)                             # works on SPARSE input
plt.scatter(coords[:, 0], coords[:, 1], c=df["cluster"], cmap="viridis", s=8)
plt.savefig("spatial_mapping.png", dpi=150)
```
- `n_clusters=4` because there are 4 newsgroups.
- `fit_predict` = fit + return the cluster id of every row.
- `cluster_centers_[c]` = the "average document" of cluster c, one weight per word → its biggest weights are what the cluster is about.
- **TruncatedSVD vs PCA:** TruncatedSVD accepts the **sparse** TF-IDF matrix directly; PCA needs `.toarray()` (dense). For thousands of posts × 4000 words, dense is wasteful.

Verified on the 10-doc class dataset: KMeans(4) put Climate Change + Economics in one cluster (top words: energy, trade, economic, emissions) — they share policy/growth vocabulary.

### Step 4 — Challenges
```python
# Challenge 1: cleaner vectorizer
custom_stops = list(ENGLISH_STOP_WORDS) + ['know','like','just','think','does','use','thanks']
clean_vec = TfidfVectorizer(stop_words=custom_stops, token_pattern=r'\b[a-zA-Z]{3,}\b', max_df=0.5)
# token_pattern: only letters, at least 3 long -> kills numbers and short noise
# max_df=0.5: drop words in more than half the posts

# Challenge 2: compare clusters with true labels
pd.crosstab(df['true_label'], df['cluster'])

# Challenge 3: length bias
lengths = df['text'].str.len()
i_long, i_short = lengths.idxmax(), lengths.idxmin()
np.linalg.norm(X_bow[i_long].toarray()), np.linalg.norm(X_bow[i_short].toarray())     # very different
np.linalg.norm(X_tfidf[i_long].toarray()), np.linalg.norm(X_tfidf[i_short].toarray()) # both 1.0

# Challenge 4: explainability
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    row = matrix[doc_id].toarray().ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = row.argsort()[::-1][:top_n]
    return [(terms[i], float(row[i])) for i in top_idx if row[i] > 0]
```
**Why Challenge 3 matters:** BoW counts grow with document length, so a 2000-word rant has a huge vector and dominates KMeans' Euclidean distances. TF-IDF rows are **L2-normalized to length 1**, so long and short posts are compared fairly. Verified: a BoW row had norm 8.43; every TF-IDF row has norm 1.0.

**Why `explain_document` returns a list of tuples:** the test checks `len(explanation[0]) == 2` → each item is `(term, weight)`.

**Why TF-IDF over an LLM in regulated settings:** every weight is traceable to a word count and a document frequency — 100% explainable, cheap, fast. A black-box model can't justify *why* a report was routed.

---

# BLOCK 9 — ★ MapReduce (mapreduce_hdfs_hadoop_repo + Lec 16/17)

### 9.1 Word count MAPPER (3 core lines)
**Mnemonic: "For line, for word, write word-TAB-1."**
```python
#!/usr/bin/env python3
import sys

def main(stream=sys.stdin, out=sys.stdout):
    for line in stream:
        for word in line.split():
            out.write(f"{word.lower()}\t1\n")

if __name__ == "__main__":
    main()
```
- Reads text from **stdin**, writes `key<TAB>value` lines to **stdout** — that's the Hadoop Streaming contract.
- Emits **`1` for every occurrence**, even repeats. The mapper does not count; it just announces "I saw this word once".
- `.lower()` → "The" and "the" become one key. `.split()` → splits on any whitespace (punctuation stays attached: `java.` ≠ `java`).

### 9.2 Word count REDUCER (the one people forget)
**Mnemonic: "Split → Changed? → Flush → Reset → Add → Last flush."**
```python
#!/usr/bin/env python3
import sys

def main(stream=sys.stdin, out=sys.stdout):
    current, total = None, 0
    for line in stream:
        key, _, value = line.rstrip("\n").partition("\t")      # SPLIT
        if key != current:                                       # CHANGED?
            if current is not None:
                out.write(f"{current}\t{total}\n")               # FLUSH previous word
            current, total = key, 0                              # RESET for new word
        total += int(value)                                      # ADD
    if current is not None:
        out.write(f"{current}\t{total}\n")                       # LAST FLUSH

if __name__ == "__main__":
    main()
```
Understand these four things and you can rebuild it from scratch:
1. **Why it works:** input arrives **sorted by key**, so all copies of a word are next to each other. The reducer only has to notice when the word **changes**.
2. **`partition("\t")`** splits at the **first** tab into `(before, tab, after)`. `_` throws away the tab.
3. **`if current is not None`** prevents writing a fake `None 0` line at the very first word.
4. **The last flush after the loop** — the final word never sees a "change", so without it **the last word is lost**. (Most common mistake.)

**If input is NOT sorted** (verified): `cat 1, dog 1, cat 1` → outputs `cat 1, dog 1, cat 1` — cat is split into two wrong lines. That's why `sort` must sit between mapper and reducer.

### 9.3 INVERTED INDEX mapper (word → which documents)
```python
def main(stream=sys.stdin, out=sys.stdout):
    for line in stream:
        doc_id, _, text = line.rstrip("\n").partition("\t")    # input: "d1<TAB>text..."
        if not doc_id:
            continue
        for word in set(text.lower().split()):                  # set() = each word ONCE per doc
            out.write(f"{word}\t{doc_id}\n")                    # emits word -> doc_id
```
Difference from wordcount: the **value is the doc_id** (not 1), and **`set()`** removes duplicates within one document.

### 9.4 INVERTED INDEX reducer
```python
def flush(word, docs, out):
    if word is not None:
        out.write(f"{word}\t{','.join(sorted(set(docs)))}\n")

def main(stream=sys.stdin, out=sys.stdout):
    current, docs = None, []
    for line in stream:
        word, _, doc_id = line.rstrip("\n").partition("\t")
        if word != current:
            flush(current, docs, out)
            current, docs = word, []
        docs.append(doc_id)
    flush(current, docs, out)
```
**Same state machine as 9.2**, only two swaps: `total = 0` → `docs = []`, and `total += int(value)` → `docs.append(doc_id)`. Output = posting list `word<TAB>d1,d2`.

### 9.5 Verified runs
`data/sample_docs.tsv` → inverted index:
```
big     d1,d2
data    d1,d2
storage d1,d3
mapreduce d2   ...
```
Lec 16/17 input (3 lines) → word count:
```
a 2   and 1   framework 2   hadoop 2   in 1   is 2   java 1
parallel 1   processing 1   simple 1   supports 1   written 1
```

### 9.6 Running it — local and on Hadoop
```bash
# Local simulation (Lec 16/17 slide):
cat input.txt | python3 mapper.py | sort | python3 reducer.py

# Repo's run_local.sh:
python3 mapper.py < input.txt | sort -k1,1 | python3 reducer.py
```
`sort` **plays the role of Shuffle & Sort**. `-k1,1` = sort on the first field only (the key).

```bash
# Real Hadoop (Lec 17 slides, older syntax)
bin/hadoop dfs -copyFromLocal /tmp/gutenberg /user/hduser/gutenberg      # 1. put input in HDFS
bin/hadoop jar hadoop-*streaming*.jar \
  -file mapper.py  -mapper mapper.py \
  -file reducer.py -reducer reducer.py \
  -input /user/hduser/gutenberg/* -output /user/hduser/gutenberg-output  # 2. run job
bin/hadoop dfs -cat /user/hduser/gutenberg-output/part-00000             # 3. read output
```
```bash
# Same thing, modern syntax (repo's run_hadoop.sh)
hdfs dfs -mkdir -p /user/$USER/wc-in
hdfs dfs -put -f input.txt /user/$USER/wc-in/
hadoop jar $STREAMING_JAR -input /user/$USER/wc-in -output /user/$USER/wc-out \
  -mapper "python3 mapper.py" -reducer "python3 reducer.py" \
  -file mapper.py -file reducer.py
hdfs dfs -cat /user/$USER/wc-out/part-*
```
**Mnemonic: "Put → Jar → Cat".** Three must-know flags: `-input`, `-output`, `-mapper/-reducer`, plus `-file` to ship the scripts to every node.
**Trap:** the **output directory must not already exist** — Hadoop refuses to overwrite it.

### 9.7 HDFS commands (from docs/01-hdfs.md)
```bash
hdfs dfs -mkdir -p /user/$USER/data          # create directory
hdfs dfs -put local.txt /user/$USER/data/    # upload
hdfs dfs -get /user/$USER/data/local.txt .   # download
hdfs dfs -ls -R /user/$USER                  # list recursively
hdfs dfs -cat /user/$USER/data/local.txt     # print contents
hdfs dfs -du -h /user/$USER                  # disk usage
hdfs dfs -setrep -w 2 /user/$USER/data/x.txt # change replication factor to 2
hdfs dfs -rm -r /user/$USER/old              # delete recursively
hdfs fsck /user/$USER -files -blocks -locations   # health + block locations
hdfs dfsadmin -report                        # capacity + live DataNodes
```
Most of them mirror Linux commands with `hdfs dfs -` in front: `mkdir, ls, cat, rm, du` — only `put`/`get` are new.

---

# BLOCK 10 — Lec 12 slide code (lower priority, but on the slides)

### 10.1 Imputation
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
num = ["age", "bmi", "glucose"]
med  = SimpleImputer(strategy="median", add_indicator=True)   # + missing flag column
knn  = KNNImputer(n_neighbors=5, weights="distance")
mice = IterativeImputer(max_iter=10, random_state=0)

med.fit(train[num])                       # fit on TRAIN only
X_tr = med.transform(train[num])
X_te = med.transform(test[num])
```

### 10.2 Outlier detection
```python
x = df["x"]
z = (x - x.mean()) / x.std()
mask_z = z.abs() > 3                                              # z-score rule

q1, q3 = x.quantile([0.25, 0.75])
iqr = q3 - q1
mask_iqr = (x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)            # IQR fence

from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.02, random_state=0)
lab = iso.fit_predict(df[["x", "y"]])                             # -1 = outlier

from sklearn.neighbors import LocalOutlierFactor
lab2 = LocalOutlierFactor(n_neighbors=20).fit_predict(df[["x", "y"]])   # density-based

df["x_w"] = x.clip(x.quantile(0.01), x.quantile(0.99))           # winsorize / cap
```

### 10.3 Aggregation in pandas
```python
cust = tx.groupby("customer_id").agg(
    n_tx=("amount", "size"),        # size counts rows (incl. NaN)
    total=("amount", "sum"),
    avg=("amount", "mean"),
    n_stores=("store_id", "nunique"),
)
monthly = tx.set_index("ts").groupby("customer_id")["amount"].resample("MS").sum()
pivot = tx.pivot_table(index="customer_id", columns="category", values="amount",
                       aggfunc="sum", fill_value=0)
win = hr.resample("30s").agg(["mean", "std", "min", "max"])   # tumbling window
roll_mean = hr.rolling("30s").mean()                          # sliding window
```
`size` counts rows; `count` ignores NaN. `resample` = groupby on time bins. `pivot_table` = one column per category.

### 10.4 Random over/under-sampling
```python
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
ros = RandomOverSampler(sampling_strategy=0.5, random_state=0)
X_o, y_o = ros.fit_resample(X_train, y_train)       # {0: 9500, 1: 500} -> {0: 9500, 1: 4750}
rus = RandomUnderSampler(sampling_strategy="majority", random_state=0)
X_u, y_u = rus.fit_resample(X_train, y_train)       # -> {0: 500, 1: 500}
```
`sampling_strategy=0.5` = target minority/majority **ratio** → 0.5 × 9500 = 4750.

### 10.5 SMOTE inside a pipeline (the "right" way)
```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)
pipe = Pipeline([
    ("scale", StandardScaler()),                      # scale BEFORE SMOTE (distances)
    ("smote", SMOTE(k_neighbors=5, random_state=42)), # runs on training folds only
    ("clf",   RandomForestClassifier(n_estimators=300, random_state=42)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["average_precision", "f1", "recall"])
```
**WRONG vs RIGHT** (the "classic mistake" slide):
```python
# WRONG: resample the whole dataset, then cross-validate -> optimistic, leaky
X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
scores = cross_val_score(clf, X_res, y_res, cv=5, scoring="f1")

# RIGHT: SMOTE inside the imblearn Pipeline -> fitted on training folds only
pipe = Pipeline([("smote", SMOTE(random_state=0)), ("clf", clf)])
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                         scoring="average_precision")
```
Use `imblearn.pipeline.Pipeline`, **not** sklearn's, because sklearn's Pipeline cannot hold a resampler.

### 10.6 SMOTE from scratch (10 lines)
```python
def smote(X_min, n_new, k=5, seed=0):
    rng = np.random.default_rng(seed)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn.kneighbors(X_min)               # idx[:, 0] is the point itself
    base = rng.integers(0, len(X_min), n_new)   # which minority points
    pick = rng.integers(1, k + 1, n_new)        # which neighbour (skip column 0 = self)
    nbr = idx[base, pick]
    lam = rng.random((n_new, 1))                # lambda ~ U(0,1)
    return X_min[base] + lam * (X_min[nbr] - X_min[base])
```
`k + 1` neighbours because the nearest neighbour of a point is **itself**.

### 10.7 Full pipeline with grid search (Lec 12 last code slide)
```python
full = ImbPipeline([
    ("pre", pre),                                   # ColumnTransformer: impute, encode, scale
    ("engineer", FunctionTransformer(add_ratio)),   # feature creation
    ("select", SelectKBest(f_classif, k=15)),       # feature selection
    ("smote", SMOTE(random_state=0)),               # balance (train only)
    ("clf", LogisticRegression(max_iter=1000)),
])
grid = {"select__k": [10, 15, 20], "smote__k_neighbors": [3, 5], "clf__C": [0.1, 1, 10]}
gs = GridSearchCV(full, grid, scoring="average_precision",
                  cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1).fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```
**Naming rule:** `stepname__parameter` with a **double underscore**. 3 × 2 × 3 = **18 combinations** × 5 folds = **90 fits**.
Pipeline order: **Impute+encode → Create → Reduce/select → Balance → Model**.

---

# QUICK-RECALL CARD (last 10 minutes before the paper)

```
TF-IDF skeleton:   from sklearn.feature_extraction.text import TfidfVectorizer
                   v = TfidfVectorizer();  X = v.fit_transform(docs)
                   v.get_feature_names_out();  X.toarray()

BoW:               CountVectorizer — same 4 lines
Similarity:        from sklearn.metrics.pairwise import cosine_similarity; cosine_similarity(X, X)
Query:             q = v.transform([query]); cosine_similarity(q, X).flatten()   # transform, NOT fit
Top terms:         X.mean(axis=0).A1  ->  argsort()[-10:][::-1]
Save/load:         from joblib import dump, load; dump(v,'f.joblib'); load('f.joblib')
Classifier:        train_test_split -> LogisticRegression(max_iter=1000).fit -> predict -> classification_report
Sparsity:          100 * (1 - X.nnz / (rows*cols))
KMeans:            KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X)
2-D map:           TruncatedSVD(n_components=2).fit_transform(X)   # sparse OK;  PCA needs .toarray()
Mapper:            for line in sys.stdin: for w in line.split(): print(f"{w.lower()}\t1")
Reducer:           split -> changed? -> flush -> reset -> add -> LAST FLUSH
Local run:         cat input.txt | mapper.py | sort | reducer.py
Hadoop:            put -> jar(-input -output -mapper -reducer -file) -> cat part-00000
sklearn IDF:       ln((1+N)/(1+df)) + 1, then L2-normalize each row
```

## Traps the examiner can set from this code
1. `"I"`, `"a"` vanish from the vocabulary (default token pattern needs ≥ 2 chars).
2. `"TF-IDF"` becomes two features: `tf`, `idf`.
3. TfidfVectorizer does **not** stem: `helps` ≠ `helpful`, `save` ≠ `saves`.
4. Calling `fit_transform` on a query instead of `transform`.
5. Reducer without the final flush → loses the last word.
6. Reducer fed unsorted input → the same word appears on several lines.
7. `cluster_map_viz.py` uses `plt` without importing it.
8. `tfidf_feature_selection.py` — the `print` in the loop is not indented.
9. Sentiment classifier fits the vectorizer before splitting → leakage; only 1 test sample.
10. `print(df.fillna(0, inplace=True))` prints `None` (pandas 2.x).
11. PCA on sparse input fails; TruncatedSVD is the sparse-friendly one.
12. Hadoop output directory must not exist.
