"""Pairwise feature engineering between a Source1 record and a candidate
record from Source2/Source3. All features are symmetric string-similarity
measures -- nothing here depends on knowing which country a record is from,
so it should generalize to France (unseen in train) without modification.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from .normalize import tokenize

FEATURE_NAMES = [
    "name_token_jaccard", "name_char3_jaccard", "name_levenshtein",
    "name_token_sort", "name_token_set", "name_partial", "name_len_diff",
    "name_exact", "name_core_exact",
    "addr_token_jaccard", "addr_char3_jaccard", "addr_levenshtein",
    "addr_token_sort", "addr_token_set", "addr_len_diff",
    "postal_match", "postal_both_present",
    "country_match",
    "blocking_score",
]

_ATTR_COLS = ("name_norm", "name_core", "addr_norm", "country", "postal")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def char_ngrams(s: str, n: int = 3) -> set:
    s = s.replace(" ", "")
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def pair_features_batch(name1_list, name2_list, name_core1_list, name_core2_list,
                         addr1_list, addr2_list, country1_list, country2_list,
                         postal1_list, postal2_list, blocking_score_list) -> "pd.DataFrame":
    """Vectorized-by-loop feature computation over PLAIN PYTHON LISTS (not a
    DataFrame with .loc lookups, and not building one dict per row -- both of
    those are the actual bottleneck at millions-of-rows scale, not the string
    similarity math itself; benchmarked at ~0.3-2us/pair this way vs.
    ~135us/pair with per-row .loc + dict construction, a >100x difference).
    Get your inputs into plain lists via a single pd.merge beforehand (see
    train.py / infer.py) -- don't call this per-row from a DataFrame loop."""
    import pandas as pd
    n = len(name1_list)
    out = {name: [0.0] * n for name in FEATURE_NAMES}

    # Cache tokenize()/char_ngrams() per UNIQUE string. With top_k candidates
    # per S1 entity, the same S1 name/address string appears ~top_k times in
    # these lists -- recomputing its token/n-gram sets each time is wasted
    # work at this scale. Memoizing cuts the dominant cost in this loop.
    tok_cache, ngram_cache = {}, {}

    def _tok(s):
        v = tok_cache.get(s)
        if v is None:
            v = tokenize(s)
            tok_cache[s] = v
        return v

    def _ngram(s):
        v = ngram_cache.get(s)
        if v is None:
            v = char_ngrams(s)
            ngram_cache[s] = v
        return v

    for i in range(n):
        name1 = name1_list[i] or ""
        name2 = name2_list[i] or ""
        addr1 = addr1_list[i] or ""
        addr2 = addr2_list[i] or ""
        name_core1 = name_core1_list[i] or ""
        name_core2 = name_core2_list[i] or ""
        postal1 = postal1_list[i] or ""
        postal2 = postal2_list[i] or ""

        n1_tok, n2_tok = _tok(name1), _tok(name2)
        a1_tok, a2_tok = _tok(addr1), _tok(addr2)

        out["name_token_jaccard"][i] = jaccard(n1_tok, n2_tok)
        out["name_char3_jaccard"][i] = jaccard(_ngram(name1), _ngram(name2))
        out["name_levenshtein"][i] = fuzz.ratio(name1, name2) / 100.0
        out["name_token_sort"][i] = fuzz.token_sort_ratio(name1, name2) / 100.0
        out["name_token_set"][i] = fuzz.token_set_ratio(name1, name2) / 100.0
        out["name_partial"][i] = fuzz.partial_ratio(name1, name2) / 100.0
        out["name_len_diff"][i] = abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)
        out["name_exact"][i] = float(bool(name1) and name1 == name2)
        out["name_core_exact"][i] = float(bool(name_core1) and name_core1 == name_core2)

        out["addr_token_jaccard"][i] = jaccard(a1_tok, a2_tok)
        out["addr_char3_jaccard"][i] = jaccard(_ngram(addr1), _ngram(addr2))
        out["addr_levenshtein"][i] = fuzz.ratio(addr1, addr2) / 100.0
        out["addr_token_sort"][i] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        out["addr_token_set"][i] = fuzz.token_set_ratio(addr1, addr2) / 100.0
        out["addr_len_diff"][i] = abs(len(addr1) - len(addr2)) / max(len(addr1), len(addr2), 1)

        out["postal_match"][i] = float(bool(postal1) and postal1 == postal2)
        out["postal_both_present"][i] = float(bool(postal1) and bool(postal2))

        out["country_match"][i] = float(country1_list[i] == country2_list[i])
        out["blocking_score"][i] = float(blocking_score_list[i])

    return pd.DataFrame(out)


def pair_features(name1, name2, name_core1, name_core2, addr1, addr2,
                   country1, country2, postal1="", postal2="",
                   blocking_score=0.0) -> dict:
    """Single-pair version -- fine for small data / the test_pipeline.py smoke
    test, but do NOT call this in a per-row loop over millions of candidate
    pairs; use pair_features_batch (or build_features_chunked below) instead."""
    name1, name2 = name1 or "", name2 or ""
    addr1, addr2 = addr1 or "", addr2 or ""
    n1_tok, n2_tok = tokenize(name1), tokenize(name2)
    a1_tok, a2_tok = tokenize(addr1), tokenize(addr2)

    return {
        "name_token_jaccard": jaccard(n1_tok, n2_tok),
        "name_char3_jaccard": jaccard(char_ngrams(name1), char_ngrams(name2)),
        "name_levenshtein": fuzz.ratio(name1, name2) / 100.0,
        "name_token_sort": fuzz.token_sort_ratio(name1, name2) / 100.0,
        "name_token_set": fuzz.token_set_ratio(name1, name2) / 100.0,
        "name_partial": fuzz.partial_ratio(name1, name2) / 100.0,
        "name_len_diff": abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1),
        "name_exact": float(bool(name1) and name1 == name2),
        "name_core_exact": float(bool(name_core1) and name_core1 == name_core2),

        "addr_token_jaccard": jaccard(a1_tok, a2_tok),
        "addr_char3_jaccard": jaccard(char_ngrams(addr1), char_ngrams(addr2)),
        "addr_levenshtein": fuzz.ratio(addr1, addr2) / 100.0,
        "addr_token_sort": fuzz.token_sort_ratio(addr1, addr2) / 100.0,
        "addr_token_set": fuzz.token_set_ratio(addr1, addr2) / 100.0,
        "addr_len_diff": abs(len(addr1) - len(addr2)) / max(len(addr1), len(addr2), 1),

        "postal_match": float(bool(postal1) and postal1 == postal2),
        "postal_both_present": float(bool(postal1) and bool(postal2)),

        "country_match": float(country1 == country2),

        "blocking_score": float(blocking_score),
    }


class IdCodec:
    """Maps entity_id strings to compact int32 positional codes and back.

    Why this matters: a real ID like "S1-0001234567" costs ~70-90 bytes as a
    Python string, and the SAME string gets duplicated once per candidate row
    (an S1 entity's ID repeats top_k times). At tens of millions of candidate
    rows that duplication is the single largest memory cost in the whole
    pipeline -- far more than the actual numeric features. An int32 code
    costs 4 bytes regardless of how many times it repeats. `codes` uses
    pandas' Index.get_indexer, which is a vectorized hash lookup (no giant
    Python dict built), and doubles as the decode table since code i IS the
    row position in the original (deduplicated) entity_id array."""
    def __init__(self, entity_ids):
        self.ids = np.asarray(entity_ids)
        self._index = pd.Index(self.ids)

    def codes(self, id_array) -> np.ndarray:
        return self._index.get_indexer(np.asarray(id_array)).astype(np.int32)

    def decode(self, codes) -> np.ndarray:
        return self.ids[codes]

    def __len__(self):
        return len(self.ids)


def build_features_chunked(candidates_df, s1p, ref_lookup, ground_truth=None,
                            chunk_size=2_000_000, verbose=True):
    """Memory-bounded feature builder for millions of candidate pairs.

    Two changes vs. a single big merge + pair_features_batch call:
      1. Entity IDs are converted to int32 codes (see IdCodec) instead of
         carrying repeated strings through the whole pipeline.
      2. Candidate pairs are processed in chunks of `chunk_size` rows at a
         time -- only one chunk's worth of actual text data is ever
         materialized at once, so peak memory doesn't grow with total
         candidate count, only with chunk_size.

    Returns:
      X            float32 ndarray (N, len(FEATURE_NAMES))
      s1_codes     int32 ndarray (N,) -- which S1 entity each row is for
      cand_codes   int32 ndarray (N,) -- which candidate record each row is for
      labels       int8 ndarray (N,) or None if ground_truth wasn't given
      s1_codec     IdCodec for decoding s1_codes back to real entity_id strings
      ref_codec    IdCodec for decoding cand_codes back to real entity_id strings
    """
    n = len(candidates_df)
    if n == 0:
        empty_X = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        s1_codec = IdCodec(s1p["entity_id"].to_numpy())
        ref_codec = IdCodec(ref_lookup["entity_id"].to_numpy())
        empty_labels = np.zeros((0,), dtype=np.int8) if ground_truth is not None else None
        return empty_X, np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), empty_labels, s1_codec, ref_codec

    s1_codec = IdCodec(s1p["entity_id"].to_numpy())
    ref_codec = IdCodec(ref_lookup["entity_id"].to_numpy())

    s1_codes_all = s1_codec.codes(candidates_df["source1_entity_id"].to_numpy())
    cand_codes_all = ref_codec.codes(candidates_df["candidate_entity_id"].to_numpy())
    blocking_scores_all = candidates_df["blocking_score"].to_numpy(dtype=np.float32)

    s1_arrs = {col: s1p[col].to_numpy() for col in _ATTR_COLS}
    ref_arrs = {col: ref_lookup[col].to_numpy() for col in _ATTR_COLS}

    gt_code_sets = None
    if ground_truth is not None:
        # every S1 entity gets an entry (empty set = singleton), matching the
        # semantics load_ground_truth/metrics.py already assume
        gt_code_sets = [set() for _ in range(len(s1_codec))]
        gt_s1_raw, gt_m_raw = [], []
        for s1_id, matches in ground_truth.items():
            for m in matches:
                gt_s1_raw.append(s1_id)
                gt_m_raw.append(m)
        if gt_s1_raw:
            gt_s1_codes = s1_codec.codes(np.asarray(gt_s1_raw))
            gt_m_codes = ref_codec.codes(np.asarray(gt_m_raw))
            for a, b in zip(gt_s1_codes, gt_m_codes):
                if a >= 0 and b >= 0:
                    gt_code_sets[a].add(b)

    feat_chunks = []
    label_chunks = [] if ground_truth is not None else None
    n_chunks = (n + chunk_size - 1) // chunk_size

    for ci, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        s1c = s1_codes_all[start:end]
        rc = cand_codes_all[start:end]
        bs = blocking_scores_all[start:end]

        feat_chunk = pair_features_batch(
            s1_arrs["name_norm"][s1c].tolist(), ref_arrs["name_norm"][rc].tolist(),
            s1_arrs["name_core"][s1c].tolist(), ref_arrs["name_core"][rc].tolist(),
            s1_arrs["addr_norm"][s1c].tolist(), ref_arrs["addr_norm"][rc].tolist(),
            s1_arrs["country"][s1c].tolist(), ref_arrs["country"][rc].tolist(),
            s1_arrs["postal"][s1c].tolist(), ref_arrs["postal"][rc].tolist(),
            bs.tolist(),
        )
        feat_chunks.append(feat_chunk[FEATURE_NAMES].to_numpy(dtype=np.float32))

        if gt_code_sets is not None:
            label_chunks.append(np.fromiter(
                (1 if b in gt_code_sets[a] else 0 for a, b in zip(s1c, rc)),
                dtype=np.int8, count=end - start,
            ))

        if verbose:
            print(f"    feature chunk {ci + 1}/{n_chunks} ({end}/{n} rows)...")
        del feat_chunk

    X = np.concatenate(feat_chunks, axis=0)
    labels = np.concatenate(label_chunks) if label_chunks is not None else None
    return X, s1_codes_all, cand_codes_all, labels, s1_codec, ref_codec
