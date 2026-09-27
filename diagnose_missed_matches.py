#!/usr/bin/env python3
r"""
Diagnoses WHY candidate recall plateaus around 78-80%, before assuming it's
a script/language problem that embeddings will fix. Runs at the project
root, next to train.py (imports from src/).

Usage (start with a sample, same pattern as train.py/run_embedding_blocking.py):

    python3 diagnose_missed_matches.py \
        --data-dir "E:\amazon ml\student_resource\dataset\train" \
        --sample-n 20000 --top-k 75

Drop --sample-n once this finishes in reasonable time on the sample.

Produces four things:
  1. Recall under your current production blocking settings (sanity check --
     should match what train.py reports).
  2. Per-country recall breakdown -- tells you whether the miss is
     concentrated in one country (India/script-related) or spread evenly
     (probably not a script problem).
  3. Per-signal RAW recall (each of your 6 blocking signals run alone,
     uncapped by top_k but still bounded by max_group_size/window like
     production) plus their union -- isolates "top_k=75 is truncating
     otherwise-found matches" from "no signal covers this pair at all".
  4. max_group_size victim analysis -- for every true match pair, checks
     whether block_key/postal/a shared name token would have connected them,
     and if so, whether that key's reference-side group was too large and
     got dropped by max_group_size. Separates "would have matched but got
     capped" from "never shared a key/token at all".
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import blocking
from src import metrics as metrics_mod

# --- Cache prepare_frame() so calling it repeatedly on the SAME DataFrame
# object (which this script does across sections 1/3/4) doesn't redo the
# name/address normalization pass over all of S2+S3 three separate times.
# This is very likely why the run looked stuck: normalizing ~10M reference
# rows three times over, silently, is slow -- not an infinite loop. ---
_prepare_frame_cache = {}
_original_prepare_frame = blocking.prepare_frame


def _cached_prepare_frame(df):
    key = id(df)
    if key not in _prepare_frame_cache:
        print(f"    (normalizing {len(df)} records -- one-time cost, cached after this)")
        _prepare_frame_cache[key] = _original_prepare_frame(df)
    return _prepare_frame_cache[key]


blocking.prepare_frame = _cached_prepare_frame


def load_sources(data_dir):
    s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t", dtype=str)
    return s1, s2, s3


def load_ground_truth(data_dir):
    path = os.path.join(data_dir, "train_ground_truth.tsv")
    gt_df = pd.read_csv(path, sep="\t", dtype=str)
    gt = {}
    for row in gt_df.itertuples(index=False):
        gt[row.source1_entity_id] = metrics_mod.parse_id_list(row.matched_entity_ids)
    return gt


def sample_s1(s1_df, ground_truth, n, seed=0):
    if n is None or n >= len(s1_df):
        return s1_df, ground_truth
    sampled_ids = s1_df["entity_id"].sample(n=n, random_state=seed)
    s1_sub = s1_df[s1_df["entity_id"].isin(sampled_ids)].reset_index(drop=True)
    gt_sub = {eid: ground_truth[eid] for eid in s1_sub["entity_id"] if eid in ground_truth}
    return s1_sub, gt_sub


# ---------------------------------------------------------------------------
# 2. Per-country recall
# ---------------------------------------------------------------------------

def recall_by_country(s1_df, candidates_df, ground_truth):
    country_of = dict(zip(s1_df["entity_id"], s1_df["country"].fillna("")))
    by_country = {}
    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue
        c = country_of.get(s1_id, "")
        by_country.setdefault(c, {})[s1_id] = true_ids

    cand_sets = candidates_df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
    report = {}
    for country, gt_sub in by_country.items():
        total_true = sum(len(v) for v in gt_sub.values())
        total_found = sum(len(v & cand_sets.get(k, set())) for k, v in gt_sub.items())
        report[country] = {
            "n_entities_with_matches": len(gt_sub),
            "total_true_matches": total_true,
            "total_found": total_found,
            "recall": total_found / total_true if total_true else float("nan"),
        }
    return report


# ---------------------------------------------------------------------------
# 3. Per-signal raw recall (isolates top_k capping from signal coverage)
# ---------------------------------------------------------------------------

def per_signal_raw_recall(s1_df, s2_df, s3_df, ground_truth, max_group_size, window,
                           same_country_only=True):
    s1p = blocking.prepare_frame(s1_df)
    signal_defs = [
        ("block_key", lambda a, b: blocking._exact_key_pairs(a, b, "block_key", max_group_size, 1.0)),
        ("postal", lambda a, b: blocking._exact_key_pairs(a, b, "postal", max_group_size, 0.9)),
        ("token_join", lambda a, b: blocking._token_join_pairs(a, b, "name_core", max_group_size)),
        ("sorted_name", lambda a, b: blocking._sorted_neighborhood_pairs(a, b, "name_core", window)),
        ("sorted_name_rev", lambda a, b: blocking._sorted_neighborhood_pairs(a, b, "name_core_rev", window)),
        ("sorted_addr", lambda a, b: blocking._sorted_neighborhood_pairs(a, b, "addr_norm", window)),
    ]
    per_signal_pairs = {name: [] for name, _ in signal_defs}

    for src_name, src_df in (("S2", s2_df), ("S3", s3_df)):
        if src_df is None or len(src_df) == 0:
            continue
        srcp = blocking.prepare_frame(src_df)
        groups = sorted(set(s1p["country"]) | set(srcp["country"])) if same_country_only else [None]
        for country in groups:
            if same_country_only:
                s1_sub = s1p[s1p["country"] == country]
                ref_sub = srcp[srcp["country"] == country]
            else:
                s1_sub, ref_sub = s1p, srcp
            if len(s1_sub) == 0 or len(ref_sub) == 0:
                continue
            for name, fn in signal_defs:
                result = fn(s1_sub, ref_sub)
                if len(result):
                    per_signal_pairs[name].append(result)

    report = {}
    union_frames = []
    for name, parts in per_signal_pairs.items():
        df = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "blocking_score"]))
        report[name] = blocking.evaluate_blocking_recall(df, ground_truth)
        union_frames.append(df[["source1_entity_id", "candidate_entity_id"]])

    union_df = (pd.concat(union_frames, ignore_index=True).drop_duplicates() if union_frames
                else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"]))
    report["UNION (uncapped by top_k, all 6 signals)"] = blocking.evaluate_blocking_recall(union_df, ground_truth)
    return report


# ---------------------------------------------------------------------------
# 4. max_group_size victim analysis
# ---------------------------------------------------------------------------

def max_group_size_victim_analysis(s1_df, s2_df, s3_df, ground_truth, max_group_size, min_token_len=3):
    s1p = blocking.prepare_frame(s1_df).set_index("entity_id")
    all_ref = pd.concat([blocking.prepare_frame(s2_df), blocking.prepare_frame(s3_df)], ignore_index=True)
    ref_by_country = {c: sub for c, sub in all_ref.groupby("country")}
    refp = all_ref.set_index("entity_id")

    key_counts_by_country = {c: sub["block_key"].value_counts() for c, sub in ref_by_country.items()}
    postal_counts_by_country = {c: sub["postal"].value_counts() for c, sub in ref_by_country.items()}

    def token_counts_for(sub):
        toks = sub["name_core"].str.split().explode()
        toks = toks[toks.str.len() >= min_token_len]
        return toks.value_counts()

    token_counts_by_country = {c: token_counts_for(sub) for c, sub in ref_by_country.items()}

    empty = pd.Series(dtype=int)
    categories = {
        "block_key": {"matched_under_cap": 0, "matched_but_DROPPED_by_cap": 0, "no_shared_key": 0},
        "postal": {"matched_under_cap": 0, "matched_but_DROPPED_by_cap": 0, "no_shared_key": 0},
        "token": {"matched_under_cap": 0, "matched_but_DROPPED_by_cap": 0, "no_shared_key": 0},
    }
    missing_ids = 0

    for s1_id, true_ids in ground_truth.items():
        if not true_ids or s1_id not in s1p.index:
            continue
        s1_row = s1p.loc[s1_id]
        country = s1_row["country"]
        for cand_id in true_ids:
            if cand_id not in refp.index:
                missing_ids += 1
                continue
            cand_row = refp.loc[cand_id]

            if s1_row["block_key"] and s1_row["block_key"] == cand_row["block_key"]:
                cnt = key_counts_by_country.get(country, empty).get(s1_row["block_key"], 0)
                key = "matched_but_DROPPED_by_cap" if cnt > max_group_size else "matched_under_cap"
                categories["block_key"][key] += 1
            else:
                categories["block_key"]["no_shared_key"] += 1

            if s1_row["postal"] and s1_row["postal"] == cand_row["postal"]:
                cnt = postal_counts_by_country.get(country, empty).get(s1_row["postal"], 0)
                key = "matched_but_DROPPED_by_cap" if cnt > max_group_size else "matched_under_cap"
                categories["postal"][key] += 1
            else:
                categories["postal"]["no_shared_key"] += 1

            s1_tokens = {t for t in s1_row["name_core"].split() if len(t) >= min_token_len}
            cand_tokens = {t for t in cand_row["name_core"].split() if len(t) >= min_token_len}
            shared = s1_tokens & cand_tokens
            if shared:
                tc = token_counts_by_country.get(country, empty)
                any_survives = any(tc.get(t, 0) <= max_group_size for t in shared)
                categories["token"]["matched_under_cap" if any_survives else "matched_but_DROPPED_by_cap"] += 1
            else:
                categories["token"]["no_shared_key"] += 1

    return categories, missing_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=75)
    ap.add_argument("--max-group-size", type=int, default=2000)
    ap.add_argument("--window", type=int, default=40)
    args = ap.parse_args()

    print("Loading data...")
    s1, s2, s3 = load_sources(args.data_dir)
    ground_truth = load_ground_truth(args.data_dir)
    print(f"  S1={len(s1)} S2={len(s2)} S3={len(s3)} ground_truth entries={len(ground_truth)}")

    if args.sample_n:
        s1, ground_truth = sample_s1(s1, ground_truth, args.sample_n)
        print(f"  sampled S1={len(s1)} ground_truth entries={len(ground_truth)}")

    print("\n=== 1. Current production-settings recall (should match train.py's number) ===")
    t0 = time.time()
    prod_candidates = blocking.generate_candidates(
        s1, s2, s3, top_k=args.top_k, max_group_size=args.max_group_size,
        window=args.window, verbose=True)
    prod_recall = blocking.evaluate_blocking_recall(prod_candidates, ground_truth)
    print(f"  {prod_recall}  ({time.time() - t0:.1f}s)")

    print("\n=== 2. Per-country recall (production settings) ===")
    for country, stats in sorted(recall_by_country(s1, prod_candidates, ground_truth).items()):
        print(f"  country={country!r:12s} recall={stats['recall']:.4f}  "
              f"({stats['total_found']}/{stats['total_true_matches']} true matches, "
              f"{stats['n_entities_with_matches']} entities with >=1 true match)")

    print("\n=== 3. Per-signal RAW recall, uncapped by top_k "
          "(still bounded by max_group_size/window like production) ===")
    t0 = time.time()
    signal_report = per_signal_raw_recall(s1, s2, s3, ground_truth, args.max_group_size, args.window)
    for name, stats in signal_report.items():
        print(f"  {name:38s} recall={stats['candidate_recall']:.4f}  "
              f"({stats['total_found']}/{stats['total_true_matches']})")
    print(f"  ({time.time() - t0:.1f}s)")
    union_recall = signal_report["UNION (uncapped by top_k, all 6 signals)"]["candidate_recall"]
    print(f"\n  Uncapped union recall:            {union_recall:.4f}")
    print(f"  Production (top_k={args.top_k}) recall:   {prod_recall['candidate_recall']:.4f}")
    print(f"  Gap attributable to top_k capping: {union_recall - prod_recall['candidate_recall']:+.4f}")
    print(f"  Remaining gap (no signal covers it at all): {1.0 - union_recall:+.4f}")

    print(f"\n=== 4. max_group_size={args.max_group_size} victim analysis ===")
    t0 = time.time()
    categories, missing = max_group_size_victim_analysis(s1, s2, s3, ground_truth, args.max_group_size)
    for sig, counts in categories.items():
        total = sum(counts.values())
        print(f"  {sig}:")
        for k, v in counts.items():
            pct = 100 * v / total if total else 0
            print(f"    {k:28s} {v:8d}  ({pct:.1f}%)")
    if missing:
        print(f"  NOTE: {missing} true-match candidate_ids were not found in S2/S3 at all "
              "(check ID formatting, or whether these reference filtered-out rows).")
    print(f"  ({time.time() - t0:.1f}s)")

    print("\n=== How to read this ===")
    print("- Section 2: if recall is similar across countries, this probably isn't primarily")
    print("  a script/language problem -- don't over-invest in embeddings before checking 3/4.")
    print("- Section 3: a big gap between union and production recall means top_k=75 is")
    print("  truncating matches your signals ALREADY find -- raising top_k (with a memory/speed")
    print("  cost you can budget) recovers this for free, no new model needed.")
    print("- Section 4: a large 'matched_but_DROPPED_by_cap' count means max_group_size is")
    print("  actively discarding true matches on an over-generic key/token -- raising it, or")
    print("  stoplisting the worst offending generic tokens instead of a blanket cutoff, should")
    print("  recover them directly.")
    print("- If a true pair shows 'no_shared_key' across ALL THREE of block_key/postal/token,")
    print("  none of your exact-key signals can ever find it regardless of caps -- that's a")
    print("  genuine case for embeddings (or a fuzzier signal), not a tuning issue.")


if __name__ == "__main__":
    main()