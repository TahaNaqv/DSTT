"""Basic unit tests for the TF-IDF activity utilities."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_data import load_documents
from src.tfidf_utils import build_tfidf, rank_documents_for_query


def test_load_documents():
    titles, docs = load_documents()
    assert len(titles) == 10
    assert len(docs) == 10
    assert all(len(d.split()) > 30 for d in docs)  # "long text"


def test_tfidf_matrix_shape():
    _, docs = load_documents()
    vectorizer, matrix = build_tfidf(docs)
    assert matrix.shape[0] == 10
    assert matrix.shape[1] == len(vectorizer.get_feature_names_out())


def test_query_ranking_matches_expected_topic():
    titles, docs = load_documents()
    vectorizer, matrix = build_tfidf(docs)
    ranking = rank_documents_for_query(
        "renewable energy and climate policy", vectorizer, matrix, titles
    )
    assert ranking[0][0] == "Climate Change"
