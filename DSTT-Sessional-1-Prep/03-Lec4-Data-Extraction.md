# Part 3 — Lecture 4: Data Extraction (+ matplotlib)

---

## 3.1 What is Data Extraction?

> **Definition: the process of retrieving data from various sources and converting it into a usable format.**
> Key concept: *"Data extraction involves extracting data from **homogeneous or heterogeneous** sources."*

**Why it matters (3 points, straight from the slide):**
1. Raw data exists in **multiple formats across disparate systems**.
2. Extraction is the **first step in ETL (Extract, Transform, Load)**.
3. **The quality of extraction determines the quality of the analysis.**

---

## 3.2 The Data Extraction Landscape

| Source | Description |
|---|---|
| **Databases** | Structured relational data |
| **APIs** | Web services, RESTful APIs |
| **Files** | CSV, JSON, XML, Excel, PDF |
| **Web Scraping** | HTML pages |
| **Sensors & IoT** | Streaming data |
| **Logs & Social Media** | Unstructured text |

**Challenges (3):**
1. **Different formats** — CSV, JSON, XML, DB tables
2. **Different structures** — structured vs unstructured
3. **Data redundancy, inconsistency, duplication**

---

## 3.3 Data Format Taxonomy (extended, 4 classes)

| Class | Property | Examples |
|---|---|---|
| **Structured** | Fixed schema, rows & columns, relational | SQL databases, CSV, Excel |
| **Semi-Structured** | Flexible/self-describing schema, hierarchical, nested objects | JSON, XML, HTML, YAML |
| **Unstructured** | No predefined structure; free text, binary, documents | Text, Images, Audio, Video, PDF, DOCX |
| **Binary Storage** | Optimized **binary serialization** | **Parquet, ORC, Avro** |

> **Key insight from the slide: extraction techniques vary significantly across these categories.**

### Extraction Considerations (4 — a likely short question)

| Consideration | Meaning |
|---|---|
| **Schema Enforcement** | **Schema-on-write vs schema-on-read** (warehouse vs lake — see Lec 8) |
| **Memory Efficiency** | **Chunking, streaming, and lazy evaluation** for files bigger than RAM |
| **Serialization Overhead** | The **CPU cost of parsing string vs binary** data |
| **Data Integrity** | Validation, missing value handling, **character encodings** |

---

## 3.4 ★ Row-Oriented vs Columnar File Formats (very testable comparison table)

| | **Row-Oriented (CSV / Excel)** | **Columnar (Parquet / Feather)** |
|---|---|---|
| **Format** | Human-readable plain text / XML compressed | **Binary columnar compression (Snappy / ZSTD)** |
| **Best use case** | Small datasets, **manual inspection, row inserts** | **Large-scale OLAP analytics and feature extraction** |
| **Bottleneck / Performance** | **The full file must be scanned to query specific columns** | **Column projection pushdown** — only load the needed columns |
| **Memory / I/O** | **High overhead during parsing** in Pandas/Python | **Predicate pushdown** filters data at the **metadata** level |

Sentence to reproduce in an exam: *"CSV stores data row by row, so reading three columns out of
fifty still costs you a full file scan; Parquet stores each column contiguously with per-column
compression and row-group statistics, so the reader can skip both unwanted columns (projection
pushdown) and unwanted row groups (predicate pushdown)."*

---

## 3.5 Semi-Structured Data

### JSON & JSONL
- Hierarchical, **key-value** data structures.
- Querying tools: **jsonpath-ng**, **pandas `json_normalize()`** (flattens nested JSON into a DataFrame).
- **JSONL (JSON Lines)**: one JSON object per line → enables **line-by-line streaming for multi-gigabyte files**.
- **Schema drift** handling: optional keys and varying nested depth.

### XML & HTML
- Tree-structured markup with **explicit tagging**.
- Querying tools: **XPath expressions, lxml, BeautifulSoup**.
- Navigation: **ElementTree** traversal — parent / child / sibling nodes.
- **Namespace handling**: registering and parsing XML namespaces.

---

## 3.6 Web Data Extraction

### RESTful APIs
- **HTTP protocols:** GET / POST with **authentication headers**.
- **Rate limiting & pagination:** handling tokens, offsets, **cursor-based pagination**.
- **Robustness:** implementing **retry logic with exponential backoff**.
- **Libraries:** **httpx** (async API extraction), **requests**.

### Web Scraping (HTML parsing)
- **Static pages:** `requests` + **BeautifulSoup4** with **CSS selectors / XPath**.
- **Dynamic rendering:** headless browser execution — **Playwright, Selenium**.
- **Best practices:** **respect `robots.txt`**, avoid IP blocking via proxy rotation.
- **Content extraction:** parsing tabular tags directly into dataframes (`pd.read_html`).

---

## 3.7 Querying Unstructured Documents

### PDF & Document Processing
- **Text extraction:** **pdfplumber, PyPDF2, pypdf**
- **Table extraction from PDFs:** pdfplumber **bounding boxes** / **camelot**
- **OCR:** **pytesseract** for scanned images/PDFs
- **Document layout parsing:** structural layout analysis (headers, captions)

### RegEx & Pattern Matching
- Extracting **entities** (emails, IP addresses, currency, dates) from raw text
- **Named Entity Recognition (NER):** **SpaCy / NLTK** for structured entity extraction
- **String normalization:** Unicode clean-up, **tokenization**, whitespace removal
- The goal: an **unstructured → structured pipeline**

---

## 3.8 Code you must be able to write

### CSV with pandas
```python
import pandas as pd
df = pd.read_csv('data.csv')

df_filtered = df[df['age'] > 30]                      # row filter (boolean mask)
df_subset   = df[['name', 'salary']]                  # column projection
avg_salary  = df.groupby('department')['salary'].mean()   # split-apply-combine
df.query('age > 25 and salary > 70000')               # query-string syntax
```

### Excel with pandas
```python
df         = pd.read_excel('data.xlsx', sheet_name='Sales')
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)   # None -> dict of ALL sheets

q1       = df[df['Quarter'] == 'Q1']
q1_sales = q1['Revenue'].sum()

with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
```
> Remember: `sheet_name=None` → **dict {sheet_name: DataFrame}**. `index=False` → don't write the index column.

---

## 3.9 Best Practices for Data Extraction (3 — likely a 3-mark question)
1. **Validate and clean data during extraction** (don't defer everything to a later stage).
2. **Handle exceptions and errors** (retries, logging, partial-failure tolerance).
3. **Use efficient querying and indexing strategies.**

---

## 3.10 Matplotlib essentials (Lecture 4 second half)

```python
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 2*np.pi, 50)
x, y = np.sin(t), np.cos(t)

plt.figure()          # create a NEW figure
plt.plot(x)
plt.figure()          # a second, separate figure
plt.plot(y)
plt.title("Cos")
```

### Format strings: `plt.plot(x, y1, 'r-o')` = colour + line style + marker

| Colour | | | Marker | |
|---|---|---|---|---|
| `b` blue · `g` green · `r` red · `c` cyan | `m` magenta · `y` yellow · `k` **black** · `w` white | | `.` point · `o` circle · `< > ^ v` triangles | `8` octagon · `s` square · `*` star · `+` plus |

> ⚠️ `k` = black (not `b`). That's a classic MCQ.

```python
plt.plot(x, y1, 'g-o', x, y2, 'b-+')
plt.legend(['sin(x)', 'sin(2x)'])
```

### Scatter with size and colour
```python
N = 50
x = np.linspace(0, 10, N)
from numpy.random import rand
e = rand(N) * 5.0            # noise
y1 = x + e
areas  = rand(N) * 300
colors = rand(N)
plt.scatter(x, y1, s=areas, c=colors)   # s = size, c = colour
plt.colorbar()
plt.title("Random scatter")
```

### Correlation matrix as an image (heatmap)
```python
corrmatrix = np.corrcoef([e1, e2, e3, e4])
plt.imshow(corrmatrix, interpolation='none', cmap='GnBu')
plt.colorbar()
```

### Subplots — `plt.subplot(nrows, ncols, index)`
```python
plt.subplot(2, 1, 1); plt.plot(x)     # top panel
plt.subplot(2, 1, 2); plt.plot(y)     # bottom panel
```

### Histograms
```python
from numpy.random import randint
data = randint(10000, size=(10, 1000))
x = np.sum(data, axis=0)              # sum of 10 uniforms -> approx NORMAL (CLT)

plt.subplot(2,1,1); plt.hist(x, color='r')
plt.subplot(2,1,2); plt.hist(x, cumulative=True)   # cumulative distribution
plt.hist([d1, d2, ...])                            # multiple histograms
```
> The `np.sum(data, axis=0)` trick demonstrates the **Central Limit Theorem**: summing many
> independent uniform variables gives an approximately **normal** distribution.

### Labels, legend, titles
```python
plt.plot(np.sin(t), label='sin')
plt.plot(np.cos(t), label='cos')
plt.legend()

plt.xlabel('radians')
plt.ylabel('amplitude', fontsize='large')   # keywords set text properties
plt.title('Sin(x)')
```
