"""
Candidate generation (blocking) for the entity resolution challenge.

REWRITTEN FOR SCALE. The original version used TF-IDF + brute-force nearest
neighbors, which is all-pairs comparison in disguise -- fine for thousands of
rows, infeasible for millions (S1~2.2M, S2~5M, S3~5.3M in this challenge).

This version uses three signals, all implemented as vectorized pandas/numpy
operations (hash joins and sorted-window scans), none of which do all-pairs
comparison:

  A. Exact block-key hash join (sorted, suffix-stripped name tokens)
     -> catches exact/near-exact renames, word-order swaps, suffix noise.
     Overly generic keys (shared by too many records) are dropped rather
     than exploding into a huge join -- see `max_group_size`.
  B. Exact postal-code hash join (within country)
     -> catches cases where the name is noisy but the postal code matches.
  C. Sorted-neighborhood window scan on the normalized name, and again on
     the normalized address -> catches typos/reorderings that don't share
     an exact key: sort the combined records by the text, and only compare
     records that land near each other in sorted order. O(n log n) sort +
     O(n * window), not O(n^2).

None of this is country-conditional logic -- country is only ever used to
partition the search space (which reduces work) or as an equality feature,
never to select different code paths. That's what keeps this correct on
France in the test set despite France never appearing in training.
"""
import numpy as np
import pandas as pd

from .normalize import (normalize_name, normalize_name_core, normalize_address,
                         extract_postal_code, block_key)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    df["name_norm"] = df["business_name"].map(normalize_name)
    df["name_core"] = df["business_name"].map(normalize_name_core)
    df["name_core_rev"] = df["name_core"].map(lambda s: s[::-1])
    df["addr_norm"] = df["business_address"].map(normalize_address)
    df["postal"] = df["business_address"].map(extract_postal_code)
    df["block_key"] = df["name_core"].map(block_key)
    return df


def check_cross_country_matches(s1_df, s2_df, s3_df, ground_truth: dict) -> int:
    """Run this once on training data. Returns count of ground-truth matches
    where the matched records have DIFFERENT country labels."""
    id_to_country = {}
    for df in (s1_df, s2_df, s3_df):
        if df is not None:
            for eid, c in zip(df["entity_id"], df["country"]):
                id_to_country[eid] = c
    cross = 0
    for s1_id, matches in ground_truth.items():
        c1 = id_to_country.get(s1_id)
        for m in matches:
            if id_to_country.get(m) != c1:
                cross += 1
    return cross


_PAIR_COLS = ["source1_entity_id", "candidate_entity_id", "blocking_score"]


def _empty_pairs():
    return pd.DataFrame(columns=_PAIR_COLS)


def _exact_key_pairs(s1_sub: pd.DataFrame, ref_sub: pd.DataFrame, key_col: str,
                      max_group_size: int = 300, score: float = 1.0) -> pd.DataFrame:
    """Vectorized hash-join blocking on an exact key. Keys shared by more than
    `max_group_size` records on the reference side are dropped -- a key that
    generic isn't selective enough to be useful, and joining on it would blow
    up combinatorially. Other signals (sorted-neighborhood, other keys) pick
    up the slack for those records."""
    left = s1_sub.loc[s1_sub[key_col] != "", ["entity_id", key_col]].rename(
        columns={"entity_id": "source1_entity_id"})
    right = ref_sub.loc[ref_sub[key_col] != "", ["entity_id", key_col]].rename(
        columns={"entity_id": "candidate_entity_id"})
    if len(left) == 0 or len(right) == 0:
        return _empty_pairs()

    counts = right[key_col].value_counts()
    ok_keys = counts[counts <= max_group_size].index
    right = right[right[key_col].isin(ok_keys)]
    if len(right) == 0:
        return _empty_pairs()

    merged = left.merge(right, on=key_col, how="inner")
    if len(merged) == 0:
        return _empty_pairs()
    merged["blocking_score"] = score
    return merged[_PAIR_COLS]


def _sorted_neighborhood_pairs(s1_sub: pd.DataFrame, ref_sub: pd.DataFrame,
                                sort_col: str, window: int = 8) -> pd.DataFrame:
    """Sort the union of both sides by `sort_col`, then pair each record with
    up to `window` neighbors in sorted order. Fully vectorized with numpy --
    no per-row Python loop over the data, so it stays fast into the millions
    of rows. Nearby sort position is used as a cheap proximity score; the
    real similarity features are computed later in features.py, not here."""
    if len(s1_sub) == 0 or len(ref_sub) == 0:
        return _empty_pairs()

    left = s1_sub[["entity_id", sort_col]].copy()
    left["_side"] = 0
    right = ref_sub[["entity_id", sort_col]].copy()
    right["_side"] = 1
    combined = pd.concat([left, right], ignore_index=True)
    combined = combined.sort_values(sort_col, kind="mergesort").reset_index(drop=True)

    ids = combined["entity_id"].to_numpy()
    sides = combined["_side"].to_numpy()
    n = len(combined)
    max_off = min(window, n - 1)
    if max_off < 1:
        return _empty_pairs()

    s1_chunks, ref_chunks, score_chunks = [], [], []
    for offset in range(1, max_off + 1):
        side_a = sides[:-offset]
        side_b = sides[offset:]
        cross = side_a != side_b
        if not cross.any():
            continue
        ids_a = ids[:-offset]
        ids_b = ids[offset:]
        s1_choice = np.where(side_a == 0, ids_a, ids_b)[cross]
        ref_choice = np.where(side_a == 0, ids_b, ids_a)[cross]
        s1_chunks.append(s1_choice)
        ref_chunks.append(ref_choice)
        score_chunks.append(np.full(cross.sum(), 1.0 / offset))

    if not s1_chunks:
        return _empty_pairs()

    return pd.DataFrame({
        "source1_entity_id": np.concatenate(s1_chunks),
        "candidate_entity_id": np.concatenate(ref_chunks),
        "blocking_score": np.concatenate(score_chunks),
    })


def _token_join_pairs(s1_sub: pd.DataFrame, ref_sub: pd.DataFrame, text_col: str = "name_core",
                       max_group_size: int = 2000, min_token_len: int = 3) -> pd.DataFrame:
    """Blocking on shared significant WORDS (not the whole name). Robust to
    scale in a way sorted-neighborhood isn't: a typo in one word of a
    multi-word name still leaves other words intact to match on.

    Score reflects HOW MANY shared tokens there are and HOW RARE they are
    (rarer/more distinctive shared words count for more, common words count
    for less), rather than a flat score for "shares at least one word". This
    matters because when top_k caps the candidate list, a true match sharing
    several distinctive words should clearly outrank a coincidental match
    sharing one common word -- with a flat score they'd tie, and which one
    survives the cap becomes arbitrary."""
    def explode_tokens(df, id_col_name):
        tmp = df[["entity_id", text_col]].copy()
        tmp["token"] = tmp[text_col].str.split()
        tmp = tmp.explode("token")
        tmp = tmp[tmp["token"].str.len() >= min_token_len]
        return tmp[["entity_id", "token"]].rename(columns={"entity_id": id_col_name})

    left = explode_tokens(s1_sub, "source1_entity_id")
    right = explode_tokens(ref_sub, "candidate_entity_id")
    if len(left) == 0 or len(right) == 0:
        return _empty_pairs()

    counts = right["token"].value_counts()
    ok_tokens = counts[counts <= max_group_size].index
    right = right[right["token"].isin(ok_tokens)]
    if len(right) == 0:
        return _empty_pairs()

    # Rarer tokens (lower count in the reference pool) are more distinctive
    # and get more weight. Log-scaled so "appears once" isn't absurdly
    # dominant over "appears a handful of times"; normalized to [0,1] per
    # token so it's comparable to the other signals' scores.
    ok_counts = counts.loc[ok_tokens]
    raw_weight = 1.0 / np.log1p(ok_counts + 1)
    token_weight = (raw_weight / raw_weight.max()).to_dict()

    merged = left.merge(right, on="token", how="inner")
    if len(merged) == 0:
        return _empty_pairs()

    merged["token_score"] = merged["token"].map(token_weight)
    agg = merged.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)["token_score"].sum()
    # Multiple shared rare tokens accumulate; cap at 1.0 to stay comparable
    # to the other signals (block_key=1.0, postal=0.9, etc).
    agg["blocking_score"] = agg["token_score"].clip(upper=1.0)
    return agg[_PAIR_COLS]


def _cap_top_k(pairs: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if len(pairs) == 0:
        return pairs
    pairs = pairs.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)["blocking_score"].max()
    pairs = pairs.sort_values(["source1_entity_id", "blocking_score"], ascending=[True, False])
    pairs["_rank"] = pairs.groupby("source1_entity_id").cumcount()
    return pairs[pairs["_rank"] < top_k].drop(columns="_rank")


def _generate_for_source(s1_sub: pd.DataFrame, ref_sub: pd.DataFrame, top_k: int,
                          window: int, max_group_size: int) -> pd.DataFrame:
    if len(s1_sub) == 0 or len(ref_sub) == 0:
        return _empty_pairs()

    parts = [
        _exact_key_pairs(s1_sub, ref_sub, "block_key", max_group_size, score=1.0),
        _exact_key_pairs(s1_sub, ref_sub, "postal", max_group_size, score=0.9),
        _token_join_pairs(s1_sub, ref_sub, "name_core", max_group_size),
        _sorted_neighborhood_pairs(s1_sub, ref_sub, "name_core", window=window),
        _sorted_neighborhood_pairs(s1_sub, ref_sub, "name_core_rev", window=window),
        _sorted_neighborhood_pairs(s1_sub, ref_sub, "addr_norm", window=window),
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return _empty_pairs()
    combined = pd.concat(parts, ignore_index=True)
    return _cap_top_k(combined, top_k)


def generate_candidates(s1_df, s2_df, s3_df, top_k=30, same_country_only=True,
                         window=40, max_group_size=2000, verbose=True) -> pd.DataFrame:
    """Returns a DataFrame: source1_entity_id, candidate_entity_id, source, blocking_score."""
    s1p = prepare_frame(s1_df)
    records = []

    for src_name, src_df in (("S2", s2_df), ("S3", s3_df)):
        if src_df is None or len(src_df) == 0:
            continue
        srcp = prepare_frame(src_df)

        if same_country_only:
            groups = sorted(set(s1p["country"]) | set(srcp["country"]))
        else:
            groups = [None]

        for country in groups:
            if same_country_only:
                s1_sub = s1p[s1p["country"] == country]
                ref_sub = srcp[srcp["country"] == country]
            else:
                s1_sub, ref_sub = s1p, srcp
            if len(s1_sub) == 0 or len(ref_sub) == 0:
                continue
            if verbose:
                print(f"    [{src_name}] country={country!r}: {len(s1_sub)} x {len(ref_sub)} records...")
            result = _generate_for_source(s1_sub, ref_sub, top_k, window, max_group_size)
            if len(result):
                result = result.copy()
                result["source"] = src_name
                records.append(result)

    if not records:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "source", "blocking_score"])
    out = pd.concat(records, ignore_index=True)
    return out[["source1_entity_id", "candidate_entity_id", "source", "blocking_score"]]


def evaluate_blocking_recall(candidates_df: pd.DataFrame, ground_truth: dict) -> dict:
    """Fraction of true matches that made it into the candidate set -- your
    recall ceiling. Run this before spending time on the classifier."""
    cand_sets = candidates_df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
    total_true = 0
    total_found = 0
    entities_with_missed = 0
    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue
        cands = cand_sets.get(s1_id, set())
        found = len(true_ids & cands)
        total_true += len(true_ids)
        total_found += found
        if found < len(true_ids):
            entities_with_missed += 1
    recall = total_found / total_true if total_true else 1.0
    return {
        "candidate_recall": recall,
        "total_true_matches": total_true,
        "total_found": total_found,
        "entities_with_missed_matches": entities_with_missed,
    }
