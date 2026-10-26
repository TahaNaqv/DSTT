# Part 1 — Lectures 1 & 2: Data, Big Data, Data Science, Tasks, Life Cycle

---

## 1.1 What is data?

> **Data are recorded observations, measurements, transactions, signals, text, images, events, or other representations.**

Three structural families — memorise these, they recur in every lecture:

| Family | Schema | Examples |
|---|---|---|
| **Structured** | Fixed schema, rows & columns, relational | SQL tables, CSV, Excel, spreadsheets |
| **Semi-structured** | Flexible, self-describing, hierarchical | JSON, XML, YAML, HTML, log files |
| **Unstructured** | No predefined structure | Free text, images, audio, video, PDFs |

Modern systems generate data *continuously and at unprecedented scale*.

### Data all around
Web & e-commerce data · financial/bank/credit transactions · online trading and purchasing · social networks.

### How much data (the slide numbers — they can be asked verbatim)
- Google processes **20 PB / day**
- Facebook: **60 TB of daily logs**
- eBay: **6.5 PB** of user data + **50 TB/day** (2009)
- 1000 Genomes project: **200 TB**
- Cost of 1 TB disk: **$35** · Time to read 1 TB at 100 MB/s: **3 hours**

The last two are the punchline: **storage is cheap, but reading/processing is the bottleneck.**
That single fact is the entire motivation for distributed computing (Hadoop/Spark, later in the course).

---

## 1.2 Big Data

> **Big Data is any data that is expensive to manage and hard to extract value from.**

Note the definition is *economic*, not a size threshold. The classic **3 V's** as taught:

| V | Meaning |
|---|---|
| **Volume** | The size of the data |
| **Velocity** | The latency of data processing relative to the growing demand for interactivity |
| **Variety & Complexity** | Diversity of sources, formats, quality, structures |

Lecture 2 adds two more (the **5 V's**): **Veracity** (trustworthiness) and **Value** (business worth).
Slide: *"Volume is only one dimension: velocity, variety, veracity, and value also matter."*

### Types of data we have (Lec 1)
Relational data (tables/transactions/legacy) · Text data (web) · Semi-structured (XML) ·
Graph data (social networks, Semantic Web/RDF) · Streaming data.

### What to do with these data?
- **Aggregation and statistics** → data warehousing and OLAP
- **Indexing, searching, querying** → keyword search, pattern matching (XML/RDF)
- **Knowledge discovery** → data mining, statistical modeling

---

## 1.3 What is Data Science?

> An area that **manages, manipulates, extracts, and interprets knowledge** from tremendous amounts of data.
> DS is a **multidisciplinary** field of study whose goal is to address the challenges in big data.
> **Data science principles apply to all data — big and small.**

Three feeding disciplines (exam-friendly triple):

| Discipline | Contributes |
|---|---|
| **Computer Science** | Pattern recognition, visualization, data warehousing, HPC, databases, AI |
| **Mathematics** | Mathematical modeling |
| **Statistics** | Statistical & stochastic modeling, probability |

**Drew Conway Venn diagram** (a slide image, commonly examined):
- Hacking Skills (Data Engineering) ∩ Math & Statistics = **Machine Learning**
- Math & Stats ∩ Substantive Expertise = **Traditional Research**
- Hacking Skills ∩ Substantive Expertise (no stats) = **Danger Zone!**
- All three = **Data Science**

**Keim's machine↔human spectrum** (other slide): Data Management, Data Mining, Machine Learning,
Statistics, Business Intelligence (machine side) ←→ Visualization ←→ Decision Making Theory,
Story Telling, Perception, Human Cognition (human side). Data Science spans the whole arrow.

**Wider ecosystem diagram:** Data Engineering, Scientific Method, Math, Statistics, Advanced
Computing, Visualization, Hacker Mindset, Domain Expertise → all point into Data Science.

### Market value quotes (they show up as fill-in-the-blank)
- *"…the most attractive job in the next 10 years will be statisticians"* — **Hal Varian, Google Chief Economist**
- McKinsey Global Institute (June 2011): the U.S. will need **140,000–190,000 predictive analysts**
  and **1.5 million managers/analysts by 2018**.
- Data Scientist = **"The Most Attractive Job of the 21st Century"**. They *find stories, extract knowledge — they are not reporters.*

### Real examples given in class
- **Target**: tracks purchases online and in-store, predictively models **which customers are going to be parents**, to market baby products to them.
- **Facebook**: asks for hometown + current location ostensibly to help friends connect, but also
  analyses these to identify **global migration patterns** and where football fanbases live.

---

## 1.4 Data Science Tasks — THE table (highest-yield table in Lecture 2)

| Task | Description | Algorithms | Example |
|---|---|---|---|
| **Classification** | Predict if a data point belongs to one of **predefined classes**, learned from a known data set | Decision Trees, Neural Networks, Bayesian models, Induction rules, **k-Nearest Neighbors** | Bucketing new customers into known customer groups |
| **Regression** | Predict the **numeric** target label of a data point | Linear regression, Logistic regression | Predicting next year's unemployment rate; estimating insurance premium |
| **Anomaly detection** | Predict if a data point is an **outlier** compared to the rest | Distance-based; Density-based | Credit-card fraud detection; network intrusion detection |
| **Time series** | Predict the target's value for a **future time frame** from history values | Exponential smoothing, **ARIMA**, regression | Sales forecasting, production forecasting, any growth phenomenon to extrapolate |
| **Clustering** | Identify **natural clusters** based on inherent properties in the data | **k-Means**, density-based clustering (**DBSCAN**) | Finding customer segments from transaction, web and call data |
| **Association analysis** | Identify **relationships within an itemset** from transaction data | **FP-Growth, Apriori** | Cross-selling opportunities for a retailer from purchase history |

Also on the "Data Science" hub diagram: Feature selection, Text mining, Deep learning, Recommendation engines.

### Two major categories

**Prediction Methods** — use some variables to predict unknown/future values of other variables.
- The attribute to be predicted = **target / dependent variable**
- The attributes used to predict = **explanatory / independent variables**

**Description Methods** — find **human-interpretable patterns** (correlations, associations, trends,
clusters, anomalies) that describe the underlying relationships in the data.

Classification as taught (memorise the bracket labels exactly — this is a classic 1-mark trap):

| Technique | Category per slide |
|---|---|
| Classification | **[Predictive]** |
| Clustering | **[Descriptive]** |
| Association Rule Discovery | **[Descriptive]** |
| Sequential Pattern Discovery (spatio-temporal) | **[Descriptive]** |
| Regression | **[Predictive]** |
| Deviation / Anomaly Detection | **[Predictive]** |

> ⚠️ Note: many textbooks put anomaly detection under descriptive. **Dr. Mateen's slide says
> [Predictive]** — answer as per the slide.

---

## 1.5 Motivating Challenges of Data Science (7 — likely a direct "list and explain" question)

1. **Scalability** — algorithms must cope with terabyte/petabyte data.
2. **High Dimensionality** — thousands of attributes; curse of dimensionality.
3. **Complex and Heterogeneous Data** — text, graphs, streams, images mixed together.
4. **Data Ownership and Distribution** — data is distributed across owners/sites; who owns it?
5. **Privacy Preservation** — analysing without exposing individuals.
6. **Streaming Data** — arrives continuously, one pass, can't store it all.
7. **Data Quality** — noise, outliers, missing values, duplicates, bias.

## 1.6 Why Data Science now? (6 drivers)
Digital transformation made data a core organizational asset · cloud platforms make large-scale
storage/compute accessible · open-source ecosystems lower the barrier to experimentation ·
ML enables prediction/recommendation/detection/automation · streaming + IoT create continuously
arriving data · **generative AI** expands how data can be queried, transformed and analysed.

> Exam-worthy line: **"More data does not automatically mean better decisions. The challenge has
> shifted from data *scarcity* toward trustworthy data *utilization*."**

## 1.7 Where is Data Science used? (domain → application)
- **Healthcare** — diagnosis support, risk prediction, medical imaging, resource planning
- **Finance** — fraud detection, credit risk, forecasting, algorithmic decision support
- **Retail** — recommendation, demand forecasting, customer segmentation
- **Manufacturing** — predictive maintenance, quality control
- **Cybersecurity** — anomaly detection, threat intelligence, behavioral analytics
- **Public sector** — planning, service delivery, policy analysis

Skills concentration for a data scientist: Mathematics & Applied Mathematics · Applied Statistics /
Data Analysis · Programming (**R, Python, Julia, SQL**) · Data Mining · Database storage & management ·
Machine Learning & discovery.

---

## 1.8 THE DATA SCIENCE LIFE CYCLE — 10 stages (near-certain exam question)

> An **end-to-end framework** for solving data-driven problems. Stages **overlap and are revisited**.
> The lifecycle is a **feedback system, not a one-way pipeline.**

| # | Stage | What happens | Key warnings from the slides |
|---|---|---|---|
| 1 | **Problem / Research Understanding** | Clarify the decision, user, stakeholder or research question. Define success *before* selecting a model. Identify constraints: latency, cost, privacy, interpretability, safety, resources. Translate the objective into measurable outcomes. | *"Reduce churn" is not yet a well-defined ML task.* |
| 2 | **Data Acquisition** | Identify internal + external sources: databases, APIs, files, sensors, logs, surveys, public datasets, streams. Check **provenance, licensing, consent, access rights, collection procedures**. | **Sampling decisions can determine the quality and representativeness of the final model.** |
| 3 | **Data Understanding** | Inspect schema, distributions, data types, ranges, duplicates, missingness, relationships. Understand *how variables were generated*. | Identify potential **leakage** and target-definition problems. EDA *begins* here and continues throughout. |
| 4 | **Data Cleaning** | Handle missing, duplicated, inconsistent, invalid, noisy records. Standardize formats and units. Resolve inconsistent categories/identifiers. Detect outliers with **domain-aware** methods. | **Cleaning decisions can change the meaning of the dataset — document them.** |
| 5 | **Exploratory Data Analysis** | Descriptive statistics + visualization. Study distributions, correlations, group differences, trends, anomalies. Look for imbalance and suspicious patterns. | **EDA should generate hypotheses, not merely produce attractive charts.** Separate exploration from confirmatory claims. |
| 6 | **Feature Engineering** | Transform raw variables into useful representations: scaling, encoding, aggregation, time-based features, text representations. Uses **domain knowledge** as well as algorithms. | **Avoid leakage: features must be available at prediction time.** A good feature can simplify a difficult learning problem. |
| 7 | **Modeling** | **Select a baseline before complex models.** Choose methods based on task, data, constraints, assumptions. Linear/logistic regression, trees, ensembles, clustering, NNs. Reproducible training & validation. | **Model complexity must be justified by measurable benefit.** |
| 8 | **Evaluation** | Metrics aligned with the *real objective*. Classification: **precision, recall, F1, ROC-AUC, PR-AUC, calibration**. Regression: **MAE, RMSE, R²**. Compare against meaningful baselines. | Evaluate **robustness, fairness, leakage, generalization** — not only headline accuracy. |
| 9 | **Deployment** | Turn the analysis/model into a usable service, application, report or decision workflow. Batch, real-time, edge, cloud or embedded. | Consider **latency, scalability, cost, security, reliability, versioning**. Deployment is where technical work meets operational reality. |
| 10 | **Monitoring & Maintenance** | Monitor data quality, **drift**, performance, latency, failures, resource use. | Define **retraining policies before failures become costly**. Track model and data versions. Monitoring closes the loop. |

### Why the lifecycle is ITERATIVE (learn these five triggers)
- New data can reveal a **flawed assumption**.
- EDA may show the **target is poorly defined**.
- Evaluation may expose **leakage or subgroup failures**.
- Deployment can reveal **latency or usability constraints**.
- Feedback creates a **new cycle of improvement**.

### "Data Science is Not Linear" — the bidirectional pairs
Problem definition ↔ data understanding · EDA ↔ feature engineering · Modeling ↔ evaluation ·
Deployment ↔ monitoring · Monitoring ↔ new data and problem reformulation.
> **"The ability to iterate is a core professional skill."**

### Shorter pipeline versions also on slides (know both)

**9-box pipeline diagram:** Problem Definition → Data Collection → Data Cleaning & Preprocessing →
EDA → Data Modeling → Model Evaluation → Reporting → Monitoring & Maintenance → Deployment (loops).

**5-step pipeline (Lec 6 & 8):**
1. Data Collection 2. Data Preparation 3. EDA 4. Machine Learning 5. Visualization & Deployment

---

## 1.9 Where the tools fit in the lifecycle (direct exam table)

| Tool | Lifecycle role |
|---|---|
| **Python / SQL** | Data access, transformation, analysis |
| **Pandas / NumPy** | Structured data manipulation and numerical computing |
| **Visualization libraries** (matplotlib, seaborn, plotly) | EDA and communication |
| **Scikit-learn / ML frameworks** | Modeling |
| **Spark** | Distributed processing |
| **Kafka** | Streaming data pipelines |
| **Cloud platforms & MLOps tools** | Deployment, orchestration, monitoring |
