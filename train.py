"""
Train the matcher on the training data.

Usage:
    python3 train.py --data-dir dataset/train --top-k 20 --out-dir artifacts

What this does:
  1. Load train_source{1,2,3}.tsv + train_ground_truth.tsv
  2. Run blocking to get candidate pairs (and report candidate recall --
     READ THIS NUMBER. If it's not >95%, improve blocking before anything
     else; no classifier can recover matches blocking never proposed.)
  3. Build features in memory-bounded CHUNKS with entity IDs represented as
     int32 codes internally, not repeated strings (see src/features.py's
     IdCodec / build_features_chunked docstrings for why this matters at
     millions-of-rows scale).
  4. Train LightGBM with GroupKFold (grouped by source1 entity CODE, so no
     entity's pairs leak across train/val) to get honest out-of-fold (OOF)
     probabilities
  5. Sweep decision thresholds against the EXACT competition metric
     (macro F_0.5, with the singleton rules) -- entirely in code-space, so
     this stays cheap even at full scale
  6. Refit on all data, save model + threshold + feature list + ID codecs
"""
import argparse
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.blocking import generate_candidates, evaluate_blocking_recall, prepare_frame, check_cross_country_matches
from src.features import build_features_chunked, FEATURE_NAMES
from src.metrics import macro_f_beta, parse_id_list


def load_ground_truth(path) -> dict:
    df = pd.read_csv(path, sep="\t", dtype=str)
    gt = {}
    for _, row in df.iterrows():
        gt[row["source1_entity_id"]] = parse_id_list(row.get("matched_entity_ids"))
    return gt


def tune_threshold_coded(s1_codes, cand_codes, oof_prob, gt_code_sets, n_s1):
    """All in code-space (ints), not strings -- cheap even at millions of
    rows. Ground truth dict is built ONCE outside the sweep; the prediction
    dict per threshold only holds entities that actually have >=1 prediction
    above that threshold (not all n_s1 entities), since macro_f_beta already
    treats a missing key as an empty (singleton) prediction."""
    gt_dict = {i: gt_code_sets[i] for i in range(n_s1)}
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.02):
        mask = oof_prob >= t
        preds = {}
        for a, b in zip(s1_codes[mask], cand_codes[mask]):
            preds.setdefault(a, set()).add(b)
        score = macro_f_beta(preds, gt_dict)
        if score > best_score:
            best_t, best_score = float(t), score
    return best_t, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/train")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=2_000_000,
                     help="Feature-building chunk size -- lower this if you still run out of RAM.")
    ap.add_argument("--sample-n", type=int, default=None,
                     help="Dry-run on only this many S1 entities (+ their ground truth) "
                          "before committing to a full run. Try --sample-n 20000 first.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    s1 = pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep="\t", dtype=str)
    ground_truth = load_ground_truth(os.path.join(args.data_dir, "train_ground_truth.tsv"))
    print(f"  S1={len(s1)} S2={len(s2)} S3={len(s3)} ground_truth entries={len(ground_truth)}")

    if args.sample_n:
        print(f"\n--sample-n {args.sample_n}: DRY RUN on a subset (S2/S3 kept full-size so "
              f"blocking behaves realistically; only S1 + its ground truth is sampled).")
        s1 = s1.sample(n=min(args.sample_n, len(s1)), random_state=0).reset_index(drop=True)
        sampled_ids = set(s1["entity_id"])
        ground_truth = {k: v for k, v in ground_truth.items() if k in sampled_ids}
        print(f"  sampled S1={len(s1)} ground_truth entries={len(ground_truth)}")

    print("\nChecking same-country assumption on ground truth...")
    cross = check_cross_country_matches(s1, s2, s3, ground_truth)
    print(f"  cross-country true matches: {cross}"
          f"{' -- LOOK INTO THIS before using same_country_only=True' if cross else ' -- safe to block by country'}")

    print("\nGenerating candidates (blocking)...")
    candidates = generate_candidates(s1, s2, s3, top_k=args.top_k, same_country_only=(cross == 0))
    print(f"  {len(candidates)} candidate pairs generated")

    recall_stats = evaluate_blocking_recall(candidates, ground_truth)
    print(f"  candidate recall ceiling: {recall_stats['candidate_recall']:.4f}  "
          f"({recall_stats['total_found']}/{recall_stats['total_true_matches']} true matches reachable)")
    if recall_stats["candidate_recall"] < 0.95:
        print("  !! Recall below 95% -- consider raising --top-k or adding an embedding-based "
              "blocking pass (see src/blocking.py docstring) before trusting the classifier step.")

    print("\nBuilding features (chunked, ID-coded)...")
    s1p = prepare_frame(s1)
    ref_lookup = pd.concat([prepare_frame(s2), prepare_frame(s3)], ignore_index=True)
    X, s1_codes, cand_codes, y, s1_codec, ref_codec = build_features_chunked(
        candidates, s1p, ref_lookup, ground_truth, chunk_size=args.chunk_size)
    print(f"  {len(X)} rows, positive rate {y.mean():.4f}, X dtype={X.dtype}, "
          f"X memory={X.nbytes / 1e9:.2f} GB")

    # code-space ground truth (needed for threshold tuning + final report).
    # Batched, not one entity at a time -- codec.codes() does a vectorized
    # hash lookup, so build the full arrays first and map once.
    gt_code_sets = [set() for _ in range(len(s1_codec))]
    gt_s1_raw, gt_m_raw = [], []
    for s1_id, matches in ground_truth.items():
        for m in matches:
            gt_s1_raw.append(s1_id)
            gt_m_raw.append(m)
    if gt_s1_raw:
        gt_s1_batch_codes = s1_codec.codes(np.asarray(gt_s1_raw))
        gt_m_batch_codes = ref_codec.codes(np.asarray(gt_m_raw))
        for a, b in zip(gt_s1_batch_codes, gt_m_batch_codes):
            if a >= 0 and b >= 0:
                gt_code_sets[a].add(b)

    print("\nTraining LightGBM with grouped CV...")
    n_groups = len(np.unique(s1_codes))
    if n_groups < 2:
        print("  !! Fewer than 2 distinct S1 entities in the candidate set -- "
              "not enough to do grouped CV. Use a larger --sample-n (or drop it "
              "for the full run).")
        return
    effective_folds = min(args.n_folds, n_groups)
    if effective_folds < args.n_folds:
        print(f"  (only {n_groups} distinct S1 entities in candidates -- "
              f"using {effective_folds}-fold CV instead of {args.n_folds}; "
              f"expected with a small --sample-n dry run)")

    gkf = GroupKFold(n_splits=max(effective_folds, 2))
    oof_prob = np.zeros(len(X), dtype=np.float32)
    models = []
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "is_unbalance": True,
        "verbosity": -1,
    }
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, s1_codes)):
        train_set = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        val_set = lgb.Dataset(X[va_idx], label=y[va_idx], reference=train_set)
        model = lgb.train(
            params, train_set, num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        oof_prob[va_idx] = model.predict(X[va_idx], num_iteration=model.best_iteration)
        models.append(model)
        print(f"  fold {fold}: best_iter={model.best_iteration}")

    print("\nTuning decision threshold against macro F_0.5 (OOF predictions, code-space)...")
    best_t, best_score = tune_threshold_coded(s1_codes, cand_codes, oof_prob, gt_code_sets, len(s1_codec))
    print(f"  BEST THRESHOLD = {best_t}   OOF macro F_0.5 = {best_score:.4f}")
    print("  (sanity: entities with zero candidates or all-below-threshold predict empty)")

    print("\nRefitting on all data for the final model...")
    train_set = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params, train_set,
        num_boost_round=int(np.mean([m.best_iteration for m in models])) or 200,
    )

    joblib.dump(final_model, os.path.join(args.out_dir, "lgbm_model.pkl"))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({
            "threshold": best_t,
            "oof_macro_f_beta": best_score,
            "top_k": args.top_k,
            "same_country_only": bool(cross == 0),
            "feature_names": FEATURE_NAMES,
            "candidate_recall": recall_stats["candidate_recall"],
        }, f, indent=2)

    # Compact NPZ instead of a giant CSV -- a CSV of the full feature table at
    # full scale would itself be tens of GB of disk (text encoding of floats
    # is very wasteful); this keeps just what's needed to inspect OOF quality.
    np.savez_compressed(
        os.path.join(args.out_dir, "oof_predictions.npz"),
        s1_codes=s1_codes, cand_codes=cand_codes, y=y, oof_prob=oof_prob,
        s1_ids=s1_codec.ids, ref_ids=ref_codec.ids,
    )

    print(f"\nSaved model + config to {args.out_dir}/")
    print("Next: python3 infer.py --data-dir dataset/test --artifacts-dir", args.out_dir)


if __name__ == "__main__":
    main()
