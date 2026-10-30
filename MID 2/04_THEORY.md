# 04 — THEORY: Lectures 9–17 (for MCQs and short answers)

Organised by theme, not by lecture, so repeated slides are learned once. Lec 9 repeats Lec 8; Lec 11 and 12 share the bias-variance and underfit/overfit slides; Lec 16 and 17 share the Hadoop-flow slides. Each section ends with the **MCQ traps**.

---

# PART A — MACHINE LEARNING FOUNDATIONS (Lec 9, 10)

## A1. Data engineering & storage (Lec 9)
- **4 concerns:** compute (Python, R, Julia, Spark, Hadoop) · storage (git, SQL, NoSQL, HBase, disk, memory) · devops (AWS, Docker, Mesos, repeatability) · product (database, web, API, viz, UI, story). *"Different at different scales."*
- **Storage ladder:** memory → disk (don't fit in memory) → cluster (still don't fit / need only parts / must bring compute to the disk).
- **Access model → database:** relational → Postgres, SQLite, HBase, VoltDB, pandas · document → MongoDB, CouchDB · key-value → Redis, Riak, Memcached · graph → Neo4J.

## A2. What ML is
- **Arthur Samuel:** the field that gives computers the ability to learn **without being explicitly programmed**; improves through experience and data.
- Traditional programming: **Data + Program → Output**. ML: **Data + Output → Program**.
- AutoML: Dataset + Optimization metric + Constraints (time/cost) → model.
- **Modern AI:** don't reverse-engineer an expert's decision process; the machine learns from training data = many (x, y) pairs. That is **supervised learning**.
- **ML algorithm:** input = training data → learns → output = a **prediction function** that gives y for x.
- **ML approach (Géron):** Study the problem → Train ML algorithm → Evaluate → (good) Launch / (bad) Analyze errors → back to study.
- **ML overall flow:** Problem nature identification → Training data collection → Model training f(x) → Model validation → Model deployment.

## A3. Supervised vs unsupervised
| | Supervised | Unsupervised |
|---|---|---|
| Data | Labeled (x, y) | Unlabeled (x) |
| Tasks | Classification, regression | Clustering / segmentation |
| Function | f: ℝᵈ→{−1,+1} (binary classifier), f: ℝᵈ→ℝ (regressor) | f: ℝᵈ→{C₁…C_k} |
| Methods | SVM, neural nets, decision trees, kNN, naïve Bayes | K-means |

- **Regression** = predicting a **continuous** output (age from a photo, house price, USD/PKR rate, steps per day).
- **Classification** = predicting a **categorical** output. **Binary** = yes/no; **multi-class** = many classes. Hard (a label) or soft/probabilistic (a probability).
- Same topic can be either: "number of steps per day" is regression; "more than 10,000 steps today?" is classification.
- **Decision boundary** separates classes (linear or curved).

## A4. Setup vocabulary (Lec 10)
- Inputs = **features**; output = **label**; d = number of features; n = number of samples; training data = (xᵢ, yᵢ), i = 1…n.
- **Training data** → train the model. **Test data** → labeled, **independent**, used only to evaluate.
- **Loss / cost / objective** = how well the model fits. **0/1 loss** for classification; **square loss** (pred − target)² for regression. Big loss = bad; zero = perfect.
- Data points are drawn from an **unknown distribution P(X, Y)**; learning = find h* = argmin L(h).
- **Process/system:** the unknown phenomenon mapping x to y; we collect training data by observing it.
- **Feature extraction / featurization** = mapping raw input to ℝᵈ → **feature vector** (= input vector). ML wants **fixed-length numeric arrays**. **Better features → less "smart" ML needed.**
- **Dimensionality reduction:** drop dimensions carrying very little information (correlated features are redundant; uncorrelated are good).

## A5. Memorizer, train/test, CV, leakage (Lec 10)
- **Memorizer:** returns the stored y for any training x, 0 otherwise → **0 training error, large error on new data** → "first glimpse of overfitting". Fix: **train/test split**.
- Intern says "1% training error, can we deploy (≤2%)?" → **No** — must do well on **new** inputs. "Don't test somebody with problems they've seen in advance."
- **Test set:** independent labeled data, e.g. **random 80/20**. Training set only for training; test set only for assessing. Bigger test set → more accurate estimate (confidence intervals).
- **Train/Test vs Train/Deploy:** the goal is that **test performance estimates deployment performance**; train/test should mirror train/deploy.
- **Time series:** random split is wrong (test days before training days). **Split in time**: train before T, test after T.
- **k-fold CV:** when the test set is too small. k equal folds; train on k−1, test on 1, rotate; report mean ± SE (SE = SD/√k). It **estimates** performance; it improves the model only **indirectly** (selection/tuning).
- **Leakage:** information about the **label sneaks into the features** (cat photos by page title; sales commission when ranking leads).

## A6. Model complexity (Lec 10, 11)
- Polynomial f(x) = w₀ + w₁x + … + w_M x^M. **Parameters** = w's (learned). **Hyperparameter** = M (set by the data scientist or a search).
- **M = 0 underfit, M = 1 underfit, M = 3 pretty good, M = 9 overfit.** M = 9 with **more data (N = 15)** → "pretty good, slightly overfit" — more data tames complexity.
- Bigger M → better training fit, **not necessarily** better test performance.
- **Hyperparameters ("tuning parameters")** control: model complexity (polynomial order), type of complexity control (L1 vs L2 regularization), optimization (learning rate), model type (loss, kernel).

---

# PART B — EVALUATION AND BIAS-VARIANCE (Lec 11)

## B1. Confusion matrix
- y = gold standard / ground truth; h(x) = predicted. Positive/Negative = what was **predicted**; True/False = whether it **matched** the truth.
- **FP = Type I error** (false alarm). **FN = Type II error** (miss).
- **Accuracy** fails on rare events (tweets 99.99%, fraud 99.83% with recall 0).
- Precision (PPV) = TP/(TP+FP); Recall = Sensitivity = TPR = TP/(TP+FN); Specificity = TNR = TN/(TN+FP); NPV = TN/(TN+FN); FNR = 1 − sensitivity ("false rejection rate"); FPR = 1 − specificity ("false acceptance rate").
- Positive likelihood ratio = TPR/FPR; negative likelihood ratio = FNR/TNR; odds = p/(1−p).
- **F1** = 2PR/(P+R) = harmonic mean; high only when **both** P and R are high.
- Coin extremes: accept 1 gold coin → precision 1.0, recall 0.1; accept all 21 → recall 1.0, precision 10/21.

## B2. Bias-variance
- **E[(Y − f̂(X))²] = Bias² + Variance + σ²** — "Total error = bias error + variance error + noise", expectation over **different possible training datasets**.
- **Bias** = how far the **average prediction** is from the truth (80 vs 100) → **underfitting**.
- **Variance** = how much the prediction **changes across training sets** → **overfitting**.
- **σ²** = irreducible noise.
- Examples: always ~70 → high bias, low variance. 95/55/110/72/40 → low bias, high variance. 82/84/81/83/82 vs 83 → both low.
- **Structural risk minimization curve:** training error falls forever; test error is **U-shaped**. Left = underfitting (high bias, low variance); right = overfitting (low bias, high variance); middle = good models.

| | Underfit | Good | Overfit |
|---|---|---|---|
| Bias | High | Low | Very low |
| Variance | Low | Low/moderate | High |
| Training error | High | Low | Very low |
| Test error | High | Low | High |
| Complexity | Too low | Appropriate | Too high |

- **Avoid overfitting:** use simple models · reduce features / feature selection · model selection · **regularization** (keep features, shrink parameters) · cross-validation to estimate test error · more training data.
- **Pre-processing is part of the model:** reduce **noise** → cleaning, smoothing, aggregation, outlier handling; reduce **variance** → fewer/better features, dimensionality reduction, balanced adequate samples; reduce **bias** → feature creation, non-linear transforms, interaction terms, better encodings. *Two pipelines with the same learner can differ more than two learners with the same pipeline.*

---

# PART C — DATA PRE-PROCESSING (Lec 12)

## C1. Pipeline & defects
- **Acquire → Clean → Integrate → Transform → Reduce → Model-ready (leak-free matrix).**
- Reduction = fewer rows (aggregation, sampling) or fewer columns (PCA/LDA/t-SNE, feature selection). Transformation = scaling, encoding, binning, feature creation. Balancing = resampling, SMOTE, class weights, threshold moving.
- **6 defects:** missing values (ICU vitals only when a nurse is present) · noise & outliers (ECG motion artifact vs real arrhythmia) · duplicates (patient under two spellings) · inconsistency (mg/dL vs mmol/L; 12h vs 24h) · imbalance & bias (fraud 0.17%) · leakage & drift (feature computed after the label event).

## C2. Missing data mechanisms ★
| | Meaning | Example | Consequence |
|---|---|---|---|
| **MCAR** | P(R\|X_obs, X_mis) = P(R) | Sensor battery dies randomly | Deletion unbiased (loses power); any imputer works |
| **MAR** | depends only on **observed** data | Older patients skip the app questionnaire; age recorded | Model-based imputation (MICE, kNN) valid |
| **MNAR** | depends on the **missing value itself** | High earners leave income blank | No purely data-driven fix: indicators, selection models, sensitivity analysis |

**Imputation:** listwise deletion (biased unless MCAR; shrinks n) · mean/median/mode (shrinks variance, weakens correlations) · missing indicator (keeps MNAR signal; more columns) · kNN (scale first; O(n²)) · iterative/MICE (strong default; slower) · native NaN (HistGradientBoosting, XGBoost). **Fit imputers on train only.**

## C3. Outliers
z-score |z| > 3 · IQR fence Q1 − 1.5·IQR, Q3 + 1.5·IQR · Mahalanobis distance (multivariate) · Isolation Forest (−1 = outlier) · LOF (density-based) · winsorize/clip at 1st–99th percentile. **Detect, then decide** — an outlier may be a real event.

## C4. Aggregation
Data reduction · change of scale (transaction → day, city → region) · stability (averages vary less) · alignment (join tables at different granularity) · privacy (group-level stats). `size` counts rows; `count` ignores NaN; `resample` = groupby on time; `pivot_table` = one column per category; `std` of a 1-row group = NaN. **Windows:** short = fast but noisy; long = stable but delayed; overlap = more but correlated windows; **split by subject, not by window** (avoid leakage).

## C5. Sampling
- Why: data too large/slow/costly to label; fast iteration. Representative sample → unbiased estimate. **Error ∝ 1/√n** (halve error → 4× data). **Representativeness beats size.**
- For training, plot a **learning curve**; if it has plateaued, more rows won't help.
- **Probability** (known inclusion probability → unbiased, error bars): simple random (with/without replacement), **systematic** (every k-th after random start; watch periodic data), **stratified** (random within each stratum; lowers variance), **cluster** (whole groups; cheap, higher variance), **multi-stage** (clusters then units).
- **Non-probability** (bias possible and unquantified): convenience, quota, purposive/judgement, snowball. Web-scraped/app data are usually convenience samples.
- **Stratify when:** strata homogeneous inside and different from each other; rare subgroups must appear; separate estimates needed; classification (**stratify on the label** so folds keep class ratios). **Never stratify on a variable unknown for the whole population.**

## C6. Imbalance and SMOTE ★
- Remedy map: **Data level** (random over/under-sampling, SMOTE, Borderline-SMOTE, ADASYN, Tomek links, ENN, NearMiss, hybrids) · **Algorithm level** (class weights, cost-sensitive loss, focal loss, threshold moving, calibration) · **Ensemble** (BalancedBagging, BalancedRandomForest, EasyEnsemble, RUSBoost, SMOTEBoost) · **Reframe** (one-class/anomaly detection, collect more minority data, change unit of analysis). **Always establish a baseline first** (unweighted model + tuned threshold).
- Random over-sampling: duplicates minority rows (no info loss; exact copies overfit). Random under-sampling: drops majority rows (fast; loses structure, high variance). Hybrid: both (two hyperparameters).
- **SMOTE:** pick minority x_i → find k (default 5) nearest **minority** neighbours → pick one x̂ → draw λ ~ U(0,1) → x_new on the segment. Stays inside the convex hull of the minority neighbourhood; fills gaps. Cost O(N_min²·d). **Needs scaled features**; numeric only (**SMOTE-NC** for mixed).
- **Family:** Borderline-SMOTE (near the boundary, "danger" points) · SVM-SMOTE (around support vectors) · **ADASYN** (more where minority is harder to learn; can boost noise) · K-Means SMOTE (inside clusters; multimodal minority) · SMOTE-NC/SMOTE-N (categorical). Plain SMOTE amplifies noise and ignores the majority.
- **Classic mistake:** resampling **before** splitting/CV → synthetic neighbours of training rows land in validation, class ratio fake → optimistic score. **Split first, resample training part only;** validation/test keep the natural ratio. Same rule for scalers, imputers, selectors, PCA.
- Score imbalanced problems with **average precision / PR-AUC, MCC, balanced accuracy**, not accuracy.
- High dimensions: Euclidean neighbours unreliable (distance concentration) → reduce dimension first or use class weights / balanced ensembles.
- **Decision guide:** mild IR < 10 → class weights + threshold, then SMOTE/Borderline · moderate–heavy numeric → SMOTE in a pipeline, then SMOTE+ENN/Tomek, ADASYN · mixed types → SMOTE-NC · very high-dim/sparse → reduce dimension or class weights · huge N extreme → under-sample + bagging, EasyEnsemble.

## C7. Full pipeline & case study
- **Impute + encode (ColumnTransformer) → Create (FunctionTransformer) → Reduce/select (PCA/SelectKBest) → Balance (SMOTE, train only) → Model.**
- **Hospital readmission:** 50,000 admissions, 55 columns, 6.2% positive. Missing labs 5–40% (**MAR**) → iterative imputer + indicators. Repeated patients → **GroupKFold on patient_id**. Diagnosis code with 700+ levels (**high cardinality**) → group into ICD chapters or CV-safe target encoding. Irregular vitals → aggregate per admission (mean, std, last, slope). Imbalance → class-weight baseline, then SMOTE inside CV. Features → filter (mutual information), then embedded (L1).
- **Pitfalls table:** fit scaler/imputer on all data → fit inside each fold · aggregate then split randomly → split by entity first · resample before CV → imblearn Pipeline · accuracy on imbalanced → PR-AUC, MCC, balanced accuracy · PCA/t-SNE on train+test → fit on train, transform the rest · select features on all data → selector inside pipeline · target-encode without CV → cross-fitted TargetEncoder · random split on grouped/time data → GroupKFold / TimeSeriesSplit.

---

# PART D — DATA DISTRIBUTIONS & PANDAS (Lec 13)

- **Distribution** = how values are spread; shows frequency/probability. Continuous → density (clusters, skew, tails); categorical → category frequencies. Formally P(X) or f(X).
- **Why:** detect patterns, skew, outliers; guides preprocessing (normalization, scaling); helps choose models.
- **Types:** normal (bell, symmetric around mean — heights, test scores; hypothesis testing) · uniform (equal probabilities) · skewed (income, house prices; **right skew = long tail on high values**, left = on low values) · multimodal (several peaks).
- **IID** = Independent and Identically Distributed: samples don't depend on each other **and** all come from the same distribution (coin tosses, **Iris**).
- **Non-IID:** not independent (time series — **stock prices** depend on previous) or not identically distributed (**federated learning across hospitals**, different demographics).
- **Why it matters:** most ML assumes IID; real data often isn't → hurts generalization, bias & fairness, harder optimization. (Links to Lec 10: non-independent time data must be split by time.)
- **Visualization:** histogram (frequency of bins), boxplot (spread, median, outliers), KDE (probability density).
- **Pandas:** built on NumPy; similar to R/MATLAB/SAS; **labeled axes avoid misalignment** when merging; handles missing/special values (−1 age, −99999 salary, NA). Two structures: **Series** (1-D, values + index, default 0…N−1) and **DataFrame** (2-D, columns of different types, row + column index).

---

# PART E — STRUCTURED vs UNSTRUCTURED DATA & TF-IDF (Lec 14)

| Type | Example | Storage | Tools |
|---|---|---|---|
| Structured | Tables, databases | SQL | Pandas, SQL |
| Semi-structured | JSON, XML | NoSQL | MongoDB, Spark |
| Unstructured | Text, images, audio | Files, cloud | Hadoop, NLP libs |

- **Structured:** fixed schema enforced on write · **ACID** transactions · fast indexed aggregations · direct ML readiness.
- **Unstructured:** no pre-defined data model; **majority** of an organization's data. Domains: social media, healthcare, multimedia, IoT, documents.
- **Characteristics:** high volume & variability · context-dependent meaning · harder to query/preprocess · often multimodal.
- **Challenges → solutions:**
  - Storage & scalability → HDFS, Amazon S3, Google Cloud Storage; MongoDB
  - Cleaning & annotation → NLTK, OpenCV, TF ImageDataGenerator; SageMaker Ground Truth
  - Feature extraction → BERT embeddings, CNN feature maps (ResNet, VGG, EfficientNet)
  - Integration into ML pipelines → Airflow, Kubeflow, MLflow; Docker, Kubernetes
- **Iceberg slide:** structured (SQL/pre-defined models) ≈ **20%** above the surface; unstructured ≈ **80%** below. Unstructured data needs rigorous **feature extraction** before ML.
- **Pipeline for unstructured data (Lec 14):** Data sources (text, images, audio, video, logs) → **Ingestion** (streaming/batch) → **Storage** (HDFS / object store / DB, + cataloging & metadata) → **Preprocessing** (cleaning / OCR / transcoding) → **Feature extraction** (TF-IDF, embeddings, CNN features) → **Modeling** (ML / DL / transfer learning, + experiment tracking: MLflow, Weights & Biases) → **Evaluation** (metrics, visuals) → **Deployment** (APIs, streaming, edge) → **Monitoring & retraining** → feedback loop back to feature extraction.
- **TF-IDF** estimates how informative a term is in one document relative to a collection. **TF** rewards terms used more in the document; **IDF** down-weights terms spread across many documents; TF-IDF highlights terms **frequent locally, rare globally**.
- **Why +1 smoothing:** keeps the denominator non-zero (no division by zero for unseen terms).
- **Applications:** search & ranking, text classification, information retrieval, similarity & feature engineering.
- sklearn differs from hand calculation: normalized TF, natural log, IDF = ln[(1+N)/(1+DF)] + 1.

---

# PART F — TEXT ANALYSIS (Lec 15)

- **Why representation matters:** raw text is unstructured, computers need numbers; representation drives performance and **interpretability**. Uses: classification, clustering, retrieval, information extraction.
- **NLP pipeline:** Text cleaning → Tokenization → Stopword removal → Stemming/Lemmatization → Feature extraction → Modeling.
- **Tokenization:** splitting document D into tokens T(D) = {w₁…w_n}.
- **Stemming** = heuristic suffix stripping (running → run, studies → **studi**); fast, low accuracy; Porter, Snowball. **Lemmatization** = morphological, dictionary form (studies → study, better → **good** as adjective); slower, high accuracy; WordNet.
- **Bag of Words:** X_ij = count of word j in doc i; **ignores grammar and word order**; sparse, high-dimensional.
- **TF-IDF:** TF × log(N/DF). **BoW counts; TF-IDF re-weights by rarity** → highlights rare but important words.
- **"Highlighter" analogy:** "The screen resolution is **stunning** but drains **battery** quickly" — TF-IDF finds the rare gems in a pile of stones; dual weighting = frequent locally, rare globally.
- **Vectorized matrix:** dark cells = high-weight, document-defining words; empty space = **sparsity**, a hallmark of TF-IDF matrices that allows fast computation.
- **Applications:** (1) extracting top features (logistics reviews: "delay", "customs" surface a regional bottleneck instead of generic "service") · (2) document similarity mapping (docs with shared high-weight words cluster spatially) · (3) topic importance per cluster (K-means on the TF-IDF matrix; TF-IDF tells *which words* define each group).
- **Why TF-IDF endures:** fast, lightweight, **fully transparent**; "~80% of the insight in 10% of the time"; deep learning/LLMs = higher accuracy but black-box and expensive — a liability in regulated industries (finance, compliance). TF-IDF = the transparent baseline.
- **"Power to make language measurable":** Standardize (tokenize, lemmatize) → Weigh (TF-IDF balances local presence with global rarity) → Visualize (spatial mapping, clustering, insights).

---

# PART G — BIG DATA, HADOOP, HDFS, MAPREDUCE (Lec 16, 17 + repo docs)

## G1. Big data definition and V's
- **Big data:** datasets whose size is beyond typical database tools' ability to capture, store, manage, analyze. The threshold rises with technology and varies by sector; today roughly dozens of TB to many PB.
- **4 V's (Lec 16):** **Volume** (data size, TB→EB) · **Velocity** (data in motion, streaming, ms–s to respond) · **Variety** (many forms: structured, unstructured, text, multimedia) · **Veracity** (data in doubt — uncertainty from inconsistency, incompleteness, ambiguity, latency, deception, model approximations). Lec 17 "big data analytics" mentions **5 V's** (+ **Value**).
- **Big data analysis pipeline:** Acquisition/Recording → Extraction/Cleaning/Annotation → Integration/Aggregation/Representation → Analysis/Modeling → Interpretation/Visualization.

## G2. Hadoop
- Open-source framework to **store and process** large datasets in a **parallel, distributed** way.
- **Three components: HDFS** (storage), **MapReduce** (processing), **YARN** (cluster resource management).
- **Cluster modes:** local (standalone), **pseudo-distributed** (single node, all daemons), fully-distributed.
- **Ecosystem:** HDFS, MapReduce, YARN at the base; Hive (SQL query), Pig (scripting), HBase (columnar store), Sqoop & Flume (data collection), Oozie (workflow), ZooKeeper (coordination), Ambari (management/monitoring), Mahout (ML), HCatalog (table/schema), Avro/Thrift (serialization/cross-language), Drill (interactive analysis).

## G3. HDFS
- Stores files as **blocks** (default **128 MB** in Hadoop 2+; 64 MB in the original GFS/MapReduce), distributed across the cluster and **replicated** (default **3**) for safety.
- **NameNode (master):** stores **metadata/namespace** (names, permissions, file → block → location), receives DataNode reports to know they're alive. Metadata files: **FsImage** = complete namespace state since NameNode start; **EditLogs** = recent changes relative to the latest FsImage. Without HA, the NameNode is a **single point of failure**. Secondary/Standby NameNode merges FsImage + EditLog.
- **DataNodes (workers):** store block replicas, serve reads/writes, send **heartbeats** and block reports.
- **Clients** ask the NameNode *where*, then read/write **directly with DataNodes** — bulk data never flows through the NameNode.
- **Rack awareness:** replica 1 on writer's node, replica 2 on a **different rack**, replica 3 on the **same rack as 2**, different node → survives node and rack failure.
- **Fault tolerance:** missed heartbeats → node dead → re-replicate its blocks; checksums detect corrupt replicas; HA with a Standby NameNode (QJM).
- Optimized for **high sequential throughput, write-once**, not low-latency random access. **Small-files problem:** each file costs NameNode memory → combine into SequenceFile/Avro/Parquet.

## G4. MapReduce
- **Motivation:** big data can't be **stored** on one machine (→ GFS/HDFS) and can't be **processed** on one machine (→ MapReduce). 20 billion pages × 20 KB = 400 TB; 30–35 MB/s → ~4 months on one machine; < 3 hours on 1000 — but you then need communication, coordination, failure recovery, status reporting, debugging, optimization, locality. MapReduce hides that.
- **Model:** map: (k1, v1) → list(k2, v2); reduce: (k2, list(v2)) → list(k3, v3).
- **Map:** grab relevant data, the user function is called per input chunk, emits (key, value) pairs to intermediate files. **Partition:** decide which of R reducers gets each key (default hash(key) mod R). **Shuffle:** reducers fetch their partition from all mappers. **Sort:** by key (different mappers send the same key). **Reduce:** called once per key with all its values; aggregates.
- **Parallelism:** maps run in parallel on different splits (programmer specifies proper input splits); reduces run in parallel on different keys; **number of reducers** is a key performance parameter. **Bottleneck: reduce can't start until map finishes.**
- **Data locality:** schedule map tasks on the machine (or rack) holding the data; inputs split in 64 MB blocks (GFS chunk size). Node-local → rack-local → off-rack.
- **Fault tolerance:** master **heartbeats** workers; failed worker's tasks are rescheduled (its completed map outputs are lost and re-run, reduce output in HDFS survives). **Master failure → whole job restarted** with a different master. **Stragglers** → speculative backup tasks.
- **Combiner:** reduce logic on each mapper's local output before shuffle → less network traffic. Valid only for **associative and commutative** ops (sum, count, max) — **not median**.
- **Partitioner:** custom routing; **hot keys** cause stragglers → key salting, range partitioning.
- **YARN:** ResourceManager (cluster resources) + ApplicationMaster (per-job coordination).
- **Patterns:** filtering (map-only), top-N, inverted index, reduce-side join, secondary sort, iterative chained jobs.
- **Limitations:** every stage writes to disk (bad for iterative/interactive), startup overhead, awkward for multi-stage/graphs/streams → motivated **Apache Spark** (in-memory).
- **Hadoop Streaming:** any executable can be mapper/reducer — read stdin, write `key<TAB>value` to stdout; reducer gets lines sorted by key.
- **Local mimic:** `cat input.txt | mapper.py | sort | reducer.py` — `sort` plays the shuffle & sort.
- **Big data datasets:** CelebFaces (1.2 GB, 200k images, 40 binary attributes) · CommonCrawl (25+ billion pages, 250 TB, semi-structured) · StackOverflow on BigQuery (182 GB).

## G5. Big data analytics (Lec 17)
- **Data** = values of qualitative/quantitative variables; **information** = meaningful/organized data; **data analytics** = examining data to draw and communicate conclusions.
- Analytics existed in the 1950s (manual spreadsheets); new tools add speed → immediate decisions → competitive edge.
- **Why important:** new opportunities, smarter decisions, efficient operations, higher revenue/profit, satisfied customers. Top 3 value drivers: **cost reduction; faster, better decisions; new products & services**.
- Scale facts: 2.5 quintillion bytes/day · 2.8 million years online in 2018 · social media 33% of online time · 2.3 billion Facebook users (2019) · ~0.5 million tweets per minute · 12-hour flight = 84 TB · Facebook 30+ PB/day > all US academic libraries (2 PB) · 500 million tweets/day.
- **6 sources:** sensors & IoT · logs & infrastructure · web & social media · transactions · scientific instruments · health records & imaging.
- **Machine-generated** = **biggest** source (sensors, GPS, satellites, IoT, Safe City video), mostly unstructured. **People-generated** = social media, mostly unstructured/semi-structured, mostly text. **Organization-generated** = students, Cricinfo, TCS tracking, government, stocks, banks, e-commerce, medical — **highly structured** (e.g., Walmart sales + Twitter sentiment).
- **Categories of data (Lec 17 slide):** structured — definite schema, stored in DB tables, ≈ **5%** of information processed · semi-structured — can't go in DB tables, metadata gives structure (XML, JSON, social network data, emails, server logs), ≈ **5–10%** · unstructured — no definite schema (social media, photos, documents, videos, CCTV, weather data), ≈ **80%**.
- **Sources (survey slide):** machine-generated, human-generated, business-generated data.
- **The analytics process (cycle):** Identifying problem → Designing data requirement → Pre-processing data → Performing analytics over data → Visualizing data.
- **Analytics tasks & methods (McKinsey chart):** ML techniques (clustering, dimensionality reduction, classification, CNN/RNN/deep nets) + other analytics (regression, search, sorting, merging, compression, graph algorithms, optimization, signal processing, encryption) → use cases (resource allocation, predictive analytics, predictive maintenance, hyper-personalization, forecasting, price optimization, converting unstructured data, triaging). Problem types: classification, prediction, generation.
- **Types of data:** relational · text (Walmart semantic search raised purchases 10–15%) · multimedia (drive-through cameras change the menu) · **time series** (equally spaced; forecasting, stock, smart meters) · sequential · **streams** (real-time, single-pass/online algorithms, irreversible decisions, small memory) · **graphs** G = (V, E) (Facebook, web graph, citation networks).
- **Challenges:** lack of talent/skills · speed of growth · data quality · compliance violations · integration complexity · security.
- **Tools:** storage (HDFS, HBase, Cassandra, MongoDB, S3) · processing (MapReduce, Spark, Flink, Storm, Dataflow) · ingestion (Kafka, NiFi, Flume, Sqoop) · querying/warehousing (Hive, Presto, Trino, BigQuery, Snowflake) · ML (Spark MLlib, H2O.ai, Mahout, TF on Spark) · visualization (Tableau, Power BI, Superset, Grafana) · cloud (AWS, GCP, Azure, Databricks) · add-ons (Delta Lake, Iceberg, Airflow, Kubeflow, MLflow).
- **Flow:** **Data → Storage → Processing → Analytics → Insight.**
- **Big data analytics** = examining large, varied datasets for hidden patterns, correlations, trends; uses the V's, distributed computing + ML; turns raw data into actionable intelligence.
- **Open research directions:** efficient learning, federated analytics, explainable analytics, edge & streaming (concept drift), data systems for AI, sustainable analytics.

---

# TOP 40 MCQ TRAPS

1. FP = **Type I**, FN = **Type II**.
2. Recall = Sensitivity = TPR; Specificity = TNR; Precision = PPV.
3. F1 is the **harmonic** mean, not arithmetic.
4. Accuracy is misleading for **rare** classes.
5. High bias ↔ underfitting; high variance ↔ overfitting.
6. Training error keeps **falling** with complexity; test error is **U-shaped**.
7. Both under- and overfitting have high **test** error; the **training** error tells them apart.
8. σ² is **irreducible**.
9. k-fold CV **estimates** performance; it doesn't directly improve the model.
10. Time series → split by **time**, not randomly.
11. Leakage = label information in the features.
12. M is a **hyperparameter**; w's are **parameters**.
13. Degree-M polynomial has **M + 1** parameters.
14. Square loss → regression; 0/1 loss → classification.
15. MCAR → deletion unbiased; MAR → model-based imputation; MNAR → no data-only fix.
16. Median is robust to outliers; mean imputation shrinks variance.
17. IQR fence uses **1.5 × IQR**; z-rule uses **|z| > 3**.
18. Isolation Forest labels outliers as **−1**.
19. SMOTE interpolates between **minority** neighbours, default **k = 5**.
20. Scale **before** SMOTE; numeric only → SMOTE-NC for mixed.
21. **Split first**, then resample the training part only.
22. Error ∝ 1/√n → halve error needs **4×** data.
23. Stratified sampling **lowers** variance; cluster sampling is cheap but **higher** variance.
24. Convenience, quota, purposive, snowball = **non-probability**.
25. Use **GroupKFold** for repeated patients; **TimeSeriesSplit** for time data.
26. IID = **independent AND identically** distributed; stock prices are **not independent**.
27. Right skew = long tail on the **high** side.
28. Stemming can produce non-words (studi); lemmatization returns dictionary words.
29. BoW ignores **grammar and word order**.
30. TF-IDF highlights words frequent **locally** but rare **globally**.
31. A word in every doc gets IDF **0** (log N/DF) but **1** in sklearn.
32. sklearn TF-IDF rows are **L2-normalized** (length 1).
33. Hadoop's 3 components: **HDFS, MapReduce, YARN**.
34. HDFS defaults: **128 MB** blocks, replication **3**.
35. NameNode stores **metadata** (FsImage + EditLogs), not data.
36. Rack awareness: 2nd replica on a **different rack**, 3rd on the **same rack as the 2nd**.
37. Reduce can't start until **map finishes**.
38. Combiner valid for sum/count/max, **not median**.
39. Master failure → **whole job restarts**; worker failure → its tasks rescheduled.
40. **Machine-generated** data is the biggest source of big data.
