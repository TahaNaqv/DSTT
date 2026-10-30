# 02 — CODE DRILLS (active recall: this is how the code actually sticks)

Reading code feels like learning but isn't. **Writing it from memory** is. These drills go in 3 levels for each block:

- **Level 1 — Fill the blanks** (most code given, key parts missing)
- **Level 2 — Skeleton only** (structure given, you write the body)
- **Level 3 — Blank page** (only the task description)

**How to use:** do Level 1 right after reading a block in `01_CODE_Memorize.md`. Do Level 2 the next session. Do Level 3 the day before the exam. Answers = `01_CODE_Memorize.md` (block number given). Write on paper, not keyboard — the exam is on paper.

Tick the box when you get it 100% right **without looking**: ☐

---

## DRILL 1 — TF-IDF skeleton (Block 4)  ★

### Level 1 ☐
```python
from sklearn.feature_extraction.______ import ______________

documents = ["Data science is fun", "Python makes data analysis easy"]

vectorizer = ______________()
X = vectorizer.______________(documents)

print(vectorizer.______________())
print(X.______())
```

### Level 2 ☐
```python
# import

# corpus of 3 documents

# create vectorizer

# learn vocabulary + IDF and build the matrix

# print feature names

# print dense matrix
```

### Level 3 ☐
> Write a program that converts 3 documents into a TF-IDF matrix, ignoring English stopwords, and prints the vocabulary, the matrix shape and the dense matrix.

---

## DRILL 2 — Bag of Words (Block 3)

### Level 1 ☐
```python
from sklearn.feature_extraction.text import ______________
corpus = ["I love data science", "Data science is amazing"]
vec = ______________()
X = vec.fit_transform(corpus)
print(vec.get_feature_names_out())
print(X.toarray())
```
**Predict the output without running it** (both lines). Then: why is "I" missing?

---

## DRILL 3 — Cosine similarity (Block 5.1)

### Level 1 ☐
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.________ import ________________

corpus = ["Machine learning enables computers to learn",
          "Deep learning is part of machine learning",
          "Natural language processing is fun"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
sim_matrix = ________________(X, X)
print(sim_matrix)
```
What is the shape of `sim_matrix`? What are the diagonal values? Why is entry [0][2] zero?

### Level 3 ☐
> Given a list `corpus`, print the cosine similarity of every document with every other document.

---

## DRILL 4 — Top terms by average TF-IDF (Block 5.2) ★

### Level 1 ☐
```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

avg_tfidf = X.mean(axis=___).___
terms = vectorizer.get_feature_names_out()
ranking = sorted(zip(________, ________), key=lambda x: x[___], reverse=______)

for term, score in ranking[:10]:
    print(f"{term:20s}: {score:.4f}")
```

### Level 2 ☐ (bar-chart version, Block 7.1)
```python
avg_scores = __________________
top_indices = avg_scores.__________()[______][______]
top_terms = [feature_names[i] for i in top_indices]
top_values = avg_scores[top_indices]
plt.______(top_terms[::-1], top_values[::-1])
```
Explain in one line each: `argsort()`, `[-10:]`, `[::-1]`.

---

## DRILL 5 — Sentiment classifier (Block 5.3)

### Level 2 ☐
```python
# 4 imports: pandas, TfidfVectorizer, train_test_split, LogisticRegression, classification_report

# DataFrame with 'text' and 'label'

# vectorize text -> X ; y = labels

# split 80/20 with random_state=42

# create LogisticRegression(max_iter=1000), fit

# predict and print classification report
```
Mnemonic check: _____ → _____ → _____ → _____

### Level 3 ☐
> Rewrite the classifier so there is **no data leakage** from the vectorizer.

---

## DRILL 6 — Save / load (Block 5.4)

### Level 1 ☐
```python
from joblib import ______, ______
vec = TfidfVectorizer()
X = vec.fit_transform(["TF-IDF saves models", "Reuse vectorizers efficiently"])
______(vec, 'vectorizer.joblib')
vec_loaded = ______('vectorizer.joblib')
X2 = vec_loaded.__________(["Save and load TF-IDF models"])
print(X2.shape)
```
What is printed? Why is it NOT `(1, 4)`?

---

## DRILL 7 — Class activity 4 functions (Block 6) ★

Mnemonic: _____ → _____ → _____ → _____

### Level 1 ☐
```python
def build_tfidf(documents, stop_words="english", ngram_range=(1, 1), max_df=1.0, min_df=1):
    vectorizer = TfidfVectorizer(stop_words=________, ngram_range=________,
                                 max_df=________, min_df=________)
    matrix = vectorizer.______________(documents)
    return ________, ________

def to_dataframe(matrix, vectorizer, titles):
    return pd.DataFrame(matrix.______(),
                        columns=vectorizer.______________(),
                        index=______)

def top_terms_per_document(df, n=5):
    result = {}
    for title in df.______:
        top = df.____[title].sort_values(ascending=______).____(n)
        result[title] = top.______()
    return result

def rank_documents_for_query(query, vectorizer, matrix, titles):
    query_vec = vectorizer.__________([query])
    scores = cosine_similarity(________, ________).__________()
    return sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)
```

### Level 3 ☐
> Write `rank_documents_for_query(query, vectorizer, matrix, titles)` that returns `(title, score)` pairs sorted from most to least similar. Then explain why `fit_transform` on the query would be wrong.

---

## DRILL 8 — Lab 15 pipeline (Block 8)

### Level 1 ☐ — cleaning
```python
tokens = [w.______() for w in word_tokenize(sample_post) if w.________()]
stop_words = ___(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w ___ ___ stop_words]
stemmed_tokens = [stemmer.____(w) for w in filtered_tokens]
lemmatized_tokens = [lemmatizer.__________(w, pos='___') for w in filtered_tokens]
```

### Level 1 ☐ — sparsity, KMeans, SVD
```python
sparsity = 100.0 * (1.0 - (X_tfidf.____ / (X_tfidf.shape[0] * X_tfidf.shape[1])))
kmeans = KMeans(n_clusters=___, random_state=42, n_init=10)
df["cluster"] = kmeans.____________(X_tfidf)
svd = ______________(n_components=2, random_state=42)
coords = svd.fit_transform(X_tfidf)
```

### Level 3 ☐
> Write `explain_document(doc_id, matrix, vectorizer, top_n=5)` returning a list of `(term, weight)` tuples for the non-zero top terms.

---

## DRILL 9 — Word count mapper + reducer (Block 9) ★★

### Level 1 ☐ — mapper
```python
import ___
for line in sys._____:
    for word in line._____():
        print(f"{word.lower()}___1")
```

### Level 1 ☐ — reducer
```python
import sys
current, total = ______, ___
for line in sys.stdin:
    key, _, value = line.rstrip("\n").__________("\t")
    if key ___ current:
        if current is not ______:
            print(f"{current}\t{total}")
        current, total = ___, ___
    total += ___(value)
if current is not None:
    print(f"{______}\t{______}")
```
Say the 6-step mnemonic out loud: _____ → _____ → _____ → _____ → _____ → _____

### Level 2 ☐ — reducer from skeleton
```python
import sys
# state variables

for line in sys.stdin:
    # split into key and value

    # if the key changed:
        # flush previous (only if one exists)
        # reset state for new key

    # add value

# final flush
```

### Level 3 ☐
> Write a Hadoop Streaming mapper and reducer in Python that count word frequencies. Then write the one-line Linux command that simulates the job locally.

---

## DRILL 10 — Inverted index (Block 9.3–9.4)

### Level 2 ☐
> Change your word-count mapper and reducer (from memory) so the output is `word<TAB>d1,d2,...`. Input lines look like `d1<TAB>some text here`.

Hints: what replaces `1` as the value? Why `set()`? What replaces `total`?

---

## DRILL 11 — Hadoop commands (Block 9.6–9.7)

### Level 1 ☐
```bash
hdfs dfs -______ -p /user/khizar/in
hdfs dfs -______ input.txt /user/khizar/in/
hadoop ___ hadoop-streaming.jar \
  -______ /user/khizar/in  -______ /user/khizar/out \
  -______ "python3 mapper.py" -______ "python3 reducer.py" \
  -____ mapper.py -____ reducer.py
hdfs dfs -____ /user/khizar/out/part-*
```
Local simulation: `cat input.txt | ________ | ____ | ________`

---

## DRILL 12 — Pandas (Block 1) ★

### Level 1 ☐
```python
data = DataFrame([[1, 2, 3], [1, NaN, NaN], [NaN, NaN, NaN], [NaN, 4, 5]])
data.dropna()                 # which rows survive? ____
data.dropna(how='____')       # drop ONLY the all-NaN row
data.dropna(axis=___, how='all')   # drop all-NaN COLUMNS
data.fillna(data.______(skipna=True))   # fill with column means
```

### Level 3 ☐ — Home Activity 1 + 2
> Create a 3×5 NumPy array of random integers 1–100, make a DataFrame with index a,b,c and columns 1–5, transpose it, set values below 40 to 0, then compute the square root of every value with a lambda.

---

## DRILL 13 — SMOTE pipeline (Block 10.5)

### Level 1 ☐
```python
from imblearn.________ import Pipeline
pipe = Pipeline([
    ("scale", ______________()),
    ("smote", ______(k_neighbors=5, random_state=42)),
    ("clf",   RandomForestClassifier(n_estimators=300, random_state=42)),
])
cv = ______________(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["______________", "f1", "recall"])
```
Why is SMOTE placed **after** the scaler? Why use `imblearn`'s Pipeline and not sklearn's?

---

## BLANK-PAGE FINAL CHECK (do this the night before) ☐

On one blank sheet, from memory, in this order, in under 25 minutes:
1. TF-IDF skeleton (I-C-F-I)
2. Query ranking with cosine similarity
3. Top-10 terms with `argsort`
4. Word-count mapper
5. Word-count reducer (with the last flush!)
6. Local pipe command + 3 Hadoop commands (Put → Jar → Cat)
7. `dropna` / `fillna` lines

If all 7 are right, you're ready for the coding section.
