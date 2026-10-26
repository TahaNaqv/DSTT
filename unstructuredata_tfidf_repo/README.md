# TF-IDF Class Activity 
A hands-on, programming-based classroom activity for teaching **TF-IDF** and **cosine similarity** using 10 long-text documents.

## 🎯 Learning Objectives
By the end of this activity, students will be able to:
- Explain Term Frequency (TF), Inverse Document Frequency (IDF), and TF-IDF.
- Build a TF-IDF matrix using `scikit-learn`.
- Interpret top-weighted terms per document.
- Rank documents against a query using cosine similarity.
- Discuss limitations of TF-IDF (no semantics, no word order).

## 🗂️ Repository Contents

| Path | Purpose |
|------|---------|
| `data/documents.py` | 10 long-text documents + titles |
| `src/load_data.py` | Loads documents into a list |
| `src/tfidf_utils.py` | Reusable TF-IDF + similarity functions |
| `src/run_activity.py` | End-to-end runnable script |
| `notebooks/tfidf_activity.ipynb` | Interactive student notebook |
| `tests/` | Basic unit tests |
| `task_to_do.txt` | Students task to be performed |

## ⚙️ Setup

### Option 1: Local (recommended)
```bash
git clone https://github.com/<your-username>/tfidf-class-activity.git
cd tfidf-class-activity
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
