"""
Offline correctness tests for embedding_blocking.py.

These do NOT download a real embedding model (no internet access needed) --
they patch embed_texts with a deterministic fake embedder built from
character n-gram hashing, so semantically/orthographically similar strings
land near each other in vector space just like a real model would, and we
can verify the FAISS indexing, top-k capping, merging, and recall-evaluation
logic actually works end-to-end. Swap in the real model (see
generate_embedding_candidates(model=...)) once you've confirmed this passes.

Run: python3 test_embedding_blocking.py
"""
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import patch

# this file lives at the project root; embedding_blocking.py lives in src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import embedding_blocking as eb


def fake_embed_texts(model, texts, batch_size=512, prefix="", show_progress=False):
    """Deterministic 64-dim bag-of-char-trigrams embedding, L2-normalized.
    Stands in for a real sentence-transformers model well enough to exercise
    the pipeline: similar strings -> similar vectors, no network needed."""
    dim = 64
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        t = (prefix + t).lower()
        if len(t) < 3:
            grams = [t] if t else []
        else:
            grams = [t[j:j + 3] for j in range(len(t) - 2)]
        for g in grams:
            out[i, hash(g) % dim] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (out / norms).astype(np.float32)


def make_data():
    s1 = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["Acme Corporation", "Globex International Pvt Ltd", "Zzy Unmatched Co"],
        "business_address": ["1 Main St, Springfield", "22 Nehru Road, Delhi", "9 Nowhere Ave"],
        "country": ["US", "India", "US"],
    })
    s2 = pd.DataFrame({
        "entity_id": ["S2-1", "S2-2", "S2-3", "S2-4"],
        "business_name": ["Acme Corp", "Globex Intl Private Limited", "Totally Different LLC", "Random Biz"],
        "business_address": ["1 Main Street, Springfield", "22 Nehru Rd, Delhi", "500 Elm St", "1 X St"],
        "country": ["US", "India", "US", "US"],
    })
    s3 = pd.DataFrame({
        "entity_id": ["S3-1"],
        "business_name": ["Acme Corporation Inc"],
        "business_address": ["1 Main St, Springfield"],
        "country": ["US"],
    })
    ground_truth = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": {"S2-2"},
        "S1-3": set(),  # singleton -- no true matches
    }
    return s1, s2, s3, ground_truth


def test_schema_and_basic_matching():
    s1, s2, s3, gt = make_data()
    with patch.object(eb, "embed_texts", side_effect=fake_embed_texts):
        out = eb.generate_embedding_candidates(
            s1, s2, s3, model="FAKE", model_name="unit-test-fake",
            fields=("name",), top_k=5, same_country_only=True,
            index_type="flat", cache_dir=None, verbose=False,
        )
    assert list(out.columns) == eb._PAIR_COLS, out.columns
    assert set(out["source"].unique()) <= {"S2", "S3"}
    assert (out["source1_entity_id"].str.startswith("S1-")).all()
    assert (out["candidate_entity_id"].str.startswith(("S2-", "S3-"))).all()

    # S1-1 "Acme Corporation" should retrieve the Acme-ish S2/S3 records
    # ahead of the unrelated ones, since the fake embedder is still
    # character-overlap-based under the hood.
    s1_1 = out[out["source1_entity_id"] == "S1-1"].sort_values("blocking_score", ascending=False)
    top_cands = set(s1_1["candidate_entity_id"].head(2))
    assert top_cands == {"S2-1", "S3-1"}, top_cands

    print("test_schema_and_basic_matching: PASS")


def test_country_partitioning_respected():
    s1, s2, s3, gt = make_data()
    with patch.object(eb, "embed_texts", side_effect=fake_embed_texts):
        out = eb.generate_embedding_candidates(
            s1, s2, s3, model="FAKE", model_name="unit-test-fake",
            fields=("name",), top_k=10, same_country_only=True,
            index_type="flat", cache_dir=None, verbose=False,
        )
    # S1-2 is India; its candidates must never include US-only S2-3/S2-4/S3-1
    s1_2_cands = set(out[out["source1_entity_id"] == "S1-2"]["candidate_entity_id"])
    assert s1_2_cands.issubset({"S2-2"}), s1_2_cands
    print("test_country_partitioning_respected: PASS")


def test_cap_top_k():
    pairs = pd.DataFrame({
        "source1_entity_id": ["S1-1"] * 5,
        "candidate_entity_id": ["S2-1", "S2-1", "S2-2", "S2-3", "S3-1"],
        "blocking_score": [0.5, 0.9, 0.8, 0.1, 0.95],
    })
    capped = eb.cap_top_k(pairs, top_k=3)
    assert len(capped) == 3
    # duplicate S2-1 rows should collapse to the max score (0.9), not sum/first
    row = capped[capped["candidate_entity_id"] == "S2-1"]
    assert abs(row["blocking_score"].iloc[0] - 0.9) < 1e-9
    assert set(capped["candidate_entity_id"]) == {"S3-1", "S2-1", "S2-2"}
    assert set(capped["source"]) == {"S3", "S2"}
    print("test_cap_top_k: PASS")


def test_merge_candidate_sources():
    string_based = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1"],
        "candidate_entity_id": ["S2-1", "S2-9"],
        "source": ["S2", "S2"],
        "blocking_score": [1.0, 0.9],
    })
    embedding_based = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1"],
        "candidate_entity_id": ["S2-1", "S3-7"],  # S2-1 overlaps, S3-7 is new
        "source": ["S2", "S3"],
        "blocking_score": [0.6, 0.99],
    })
    merged = eb.merge_candidate_sources(string_based, embedding_based, top_k=10)
    assert set(merged["candidate_entity_id"]) == {"S2-1", "S2-9", "S3-7"}
    row = merged[merged["candidate_entity_id"] == "S2-1"]
    # overlapping pair should keep the MAX score across sources (1.0, not 0.6)
    assert abs(row["blocking_score"].iloc[0] - 1.0) < 1e-9
    print("test_merge_candidate_sources: PASS")


def test_recall_uplift_on_toy_data():
    """Simulates the exact workflow you'd run for real: generate embedding
    candidates, evaluate recall the same way blocking.py does, and confirm
    embedding candidates can recover a true match that a naive exact-token
    signal would have missed (here: 'Acme Corporation' vs 'Acme Corp',
    already fine for string blocking too, but the mechanism generalizes to
    cross-script pairs once you swap in a real multilingual model).

    Assumes this test file sits next to your src/ package (same layout as
    this file ships in) so `src.blocking` imports cleanly via its package's
    own relative imports. Adjust the path insert if your layout differs."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.blocking import evaluate_blocking_recall

    s1, s2, s3, gt = make_data()
    with patch.object(eb, "embed_texts", side_effect=fake_embed_texts):
        out = eb.generate_embedding_candidates(
            s1, s2, s3, model="FAKE", model_name="unit-test-fake",
            fields=("name",), top_k=5, same_country_only=True,
            index_type="flat", cache_dir=None, verbose=False,
        )
    metrics = evaluate_blocking_recall(out, gt)
    assert metrics["candidate_recall"] == 1.0, metrics
    print(f"test_recall_uplift_on_toy_data: PASS ({metrics})")


if __name__ == "__main__":
    test_schema_and_basic_matching()
    test_country_partitioning_respected()
    test_cap_top_k()
    test_merge_candidate_sources()
    test_recall_uplift_on_toy_data()
    print("\nAll tests passed.")
