#!/usr/bin/env python3
r"""
Sweeps (top_k, max_group_size) combos WITHOUT re-normalizing S2/S3 for each
one -- normalization (the ~7min-at-full-scale cost) happens once, cached,
then each combo only re-runs the actual join/scan signals. Use this instead
of rerunning diagnose_missed_matches.py per config.

Based on your diagnostic results: top_k=75 truncation alone costs you 12.9
recall points (union=0.9135 vs production=0.7845), and max_group_size=2000
drops 34.4% of token-join's true matches. This sweeps a small, deliberately
short grid around those two knobs so you get an empirical answer instead of
more theorizing, in well under the time it'd take to re-run the diagnostic
per config.

    python3 sweep_blocking_params.py \
        --data-dir "E:\amazon ml\student_resource\dataset\train" --sample-n 20000

Sweeps top_k, max_group_size, AND window (all 3 blocking caps) -- by default,
6 hand-picked combos that isolate each lever individually before trying them
combined, rather than a full cross-product (which would re-pay join-computation
cost per combo and take much longer for not much extra information). Pass
--combos "75:2000:40,150:10000:80" to specify your own top_k:max_group_size:window
triples instead.
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import blocking
from src import metrics as metrics_mod

_prepare_frame_cache = {}
_original_prepare_frame = blocking.prepare_frame


def _cached_prepare_frame(df):
    key = id(df)
    if key not in _prepare_frame_cache:
        t0 = time.time()
        _prepare_frame_cache[key] = _original_prepare_frame(df)
        print(f"    (normalized {len(df)} records in {time.time() - t0:.1f}s -- cached for the rest of this run)")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=20000)
    ap.add_argument("--combos", default=None,
                     help="comma-separated top_k:max_group_size:window triples, e.g. "
                          "'75:2000:40,150:10000:80'. If omitted, uses a default set that "
                          "isolates each of the 3 levers individually, then tries them combined.")
    args = ap.parse_args()

    if args.combos:
        combos = []
        for triple in args.combos.split(","):
            k, mgs, w = triple.split(":")
            combos.append((int(k), int(mgs), int(w)))
    else:
        # (top_k, max_group_size, window) -- baseline first, then isolate each
        # lever, then two combined bets.
        combos = [
            (75, 2000, 40),      # baseline -- known: recall=0.7845
            (150, 2000, 40),     # isolate top_k
            (75, 10000, 40),     # isolate max_group_size
            (75, 2000, 80),      # isolate window
            (150, 10000, 80),    # combined, moderate
            (250, 10000, 80),    # combined, aggressive
        ]

    print("Loading data...")
    s1, s2, s3 = load_sources(args.data_dir)
    ground_truth = load_ground_truth(args.data_dir)
    if args.sample_n:
        s1, ground_truth = sample_s1(s1, ground_truth, args.sample_n)
    print(f"  S1={len(s1)} (sampled) S2={len(s2)} S3={len(s3)} ground_truth entries={len(ground_truth)}")

    full_s1_count = 2_206_821  # your known full-scale S1 size, for extrapolation
    scale_factor = full_s1_count / len(s1)

    print(f"\nTrying {len(combos)} (top_k, max_group_size, window) combos. "
          f"Normalization happens once total, not per combo.\n")

    results = []
    for k, mgs, w in combos:
        print(f"--- top_k={k}, max_group_size={mgs}, window={w} ---")
        t0 = time.time()
        candidates = blocking.generate_candidates(
            s1, s2, s3, top_k=k, max_group_size=mgs, window=w, verbose=False)
        dt = time.time() - t0
        recall = blocking.evaluate_blocking_recall(candidates, ground_truth)
        n_cand = len(candidates)
        est_full_cand = int(n_cand * scale_factor)
        print(f"  recall={recall['candidate_recall']:.4f}  "
              f"({recall['total_found']}/{recall['total_true_matches']})  "
              f"candidates={n_cand}  est.full-scale-candidates=~{est_full_cand:,}  "
              f"({dt:.1f}s)")
        results.append({
            "top_k": k, "max_group_size": mgs, "window": w, "recall": recall["candidate_recall"],
            "n_candidates_sample": n_cand, "est_full_scale_candidates": est_full_cand,
            "seconds": round(dt, 1),
        })

    print("\n=== Summary (sorted by recall) ===")
    df = pd.DataFrame(results).sort_values("recall", ascending=False)
    print(df.to_string(index=False))

    print("\n=== How to read this ===")
    print("- Compare each row's 'est_full_scale_candidates' against what your pipeline already")
    print("  handled successfully: your original top_k=75 sample run produced 3,000,000 candidate")
    print("  rows for a 20k sample -- that's your known-working reference point for full-scale")
    print("  feature-building/training time and memory.")
    print("- The 3 single-lever rows (vs baseline) tell you which of top_k/max_group_size/window")
    print("  is actually doing the work, so you're not paying compute for a lever that isn't")
    print("  moving recall on your data.")
    print("- Pick the config with the best recall whose est_full_scale_candidates you're willing")
    print("  to commit compute time to tonight, then rerun train.py with that top_k/max_group_size/window.")


if __name__ == "__main__":
    main()
