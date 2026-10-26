> ## ⚠️ SUPERSEDED IN PART — READ `08-PAST-PAPERS-ANALYSIS.md` FIRST
>
> The two past papers (Fall-2024 and Fall-2025) contain **zero theory/definition questions**.
> Both are 100% applied: dirty-table diagnosis, encoding decisions, pandas code, and (2025) a
> research-paper comprehension question worth **40% of the marks**.
> The "likely question formats" section below was written before the past papers were available;
> treat files 01–05 as background knowledge, and **drill file 08 as the actual target.**

# Sessional-I — Exam Map (DS-5002 Data Science Tools & Techniques)

Instructor: Dr. M. Mateen Yaqoob · Weight: 10–12% · Covers Lectures 1–8 (Weeks 1–3+)

## What has actually been taught (= your scope)

| Lec | Week | Topic | Exam weight |
|---|---|---|---|
| 1 | 1 | What is data, Big Data, what is Data Science | Medium |
| 2 | 1 | DS tasks, predictive vs descriptive, DS life cycle (10 stages) | **HIGH** |
| 3 | 2 | Data & attribute types, data quality, preprocessing, sampling, dim. reduction | **HIGHEST** |
| 4 | 2 | Data extraction (formats, row vs columnar, JSON/XML, API/scraping, PDF), feature subset selection, attribute transformation, matplotlib | **HIGH** |
| 5 | 3 | Data preparation, the Adult dataset, data scrubbing | Medium |
| 6 | 3 | Data wrangling, filtering, preprocessing pipeline, feature engineering, integration, dim. reduction | **HIGH** |
| 7 | — | EDA: patterns, EDA vs CDA, data reduction, visualization do's/don'ts | **HIGH** |
| 8 | — | EDA techniques tables, data warehouse vs lake, data engineering/storage, Intro to ML | **HIGH** |

Plus: **Activity-1** (Adult dataset cleaning) and **Assignment-1** (Census-Income KDD + attached CSV)
— the instructor writes exam questions from these. Papers are mostly Sessional-II / final material,
but "structured vs unstructured data" from the Unstructured Data paper is fair game.

## Course outline mapping (from the official form)

Introduction (3) → Data Extraction (6) → Data Scrubbing (6) → Python DS Stack & Wrangling (6)
→ Data Storage & EDA (6) → Data Modelling (6) → Model Evaluation (3) → Big Data Tools (6) → Recent Trends (6)

Sessional-I sits after "Data Storage and Exploratory Analysis" has begun and ML has just been
introduced. So: **everything up to and including intro-to-ML is examinable; model evaluation
metrics, MapReduce/Hadoop and ethics are NOT yet** (Sessional-II).

## Likely question formats

1. **Define + differentiate** (nominal vs ordinal vs interval vs ratio; EDA vs CDA; normalization vs standardization; data warehouse vs data lake; supervised vs unsupervised; outlier vs error).
2. **Classify this attribute** — they give a table of columns, you assign the measurement level / type.
3. **Given a dirty table, list the issues and your fix** (the Lec-5 slide-4 dirty table is a *very* likely exam table).
4. **Explain the DS life cycle** and why it is iterative.
5. **Given a scenario, pick the DS task** (classification / regression / clustering / anomaly / time series / association) and name an algorithm.
6. **Given an objective/data type, pick the EDA technique** (Lec-8 tables).
7. **Short code reading/writing in pandas** (read_csv, filtering, groupby, fillna, drop_duplicates, ColumnTransformer/Pipeline).
8. **Justify a preprocessing decision with evidence** — the Activity-1 style: Problem → Evidence → Action → Justification.
