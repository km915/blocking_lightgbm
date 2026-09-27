#!/usr/bin/env python3
"""
CLI driver for embedding-based blocking. Mirrors the dry-run pattern you're
already using in train.py (--sample-n keeps S2/S3 full-size and only samples
S1 + its ground truth, so blocking behaves realistically).

Typical usage while validating (fast, small model, name field only):

    python3 run_embedding_blocking.py \\
        --data-dir "E:\\amazon ml\\student_resource\\dataset\\train" \\
        --sample-n 20000 --top-k 75 --fields name

Once that looks good, drop --sample-n to run on the full training set, and
check candidate_recall the same way train.py's blocking step does. If you
already have string-based candidates saved (from blocking.generate_candidates,
pickled as a DataFrame), pass --existing-candidates to see the MERGED recall,
which is the number that actually matters for the classifier stage.

Assumes this script sits at the same level as your `src/` package (the one
containing blocking.py, features.py, metrics.py, normalize.py, and this
repo's embedding_blocking.py, which you should copy into src/ alongside
them). Adjust SRC_DIR below if your layout differs.
"""
import argparse
import os
import sys
import time

import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from src import blocking as string_blocking          # noqa: E402
from src import metrics as metrics_mod                # noqa: E402
from src import embedding_blocking as eb              # noqa: E402


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
    sampled_ids = s1_df["entity_id"].sample(n=min(n, len(s1_df)), random_state=seed)
    s1_sub = s1_df[s1_df["entity_id"].isin(sampled_ids)].reset_index(drop=True)
    gt_sub = {eid: ground_truth[eid] for eid in s1_sub["entity_id"] if eid in ground_truth}
    return s1_sub, gt_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-name", default="intfloat/multilingual-e5-small",
                     help="MIT: intfloat/multilingual-e5-small (fast, default). "
                          "Apache-2.0: sentence-transformers/LaBSE (slower, "
                          "purpose-built for cross-lingual bitext retrieval).")
    ap.add_argument("--fields", default="name",
                     help="comma-separated: name, address, name_address. "
                          "Start with 'name' only -- it's the primary pain point "
                          "and half the encoding cost of adding address too.")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default=None, help="cuda / cpu / mps; auto-detects if omitted")
    ap.add_argument("--index-type", default="auto", choices=["auto", "flat", "ivfflat", "ivfpq"])
    ap.add_argument("--cache-dir", default="./emb_cache")
    ap.add_argument("--sample-n", type=int, default=None,
                     help="dry run: sample this many S1 entities (S2/S3 stay full-size)")
    ap.add_argument("--existing-candidates", default=None,
                     help="optional pickle of a DataFrame from blocking.generate_candidates "
                          "(source1_entity_id, candidate_entity_id, source, blocking_score) "
                          "to report MERGED recall against, not just embedding-alone recall")
    ap.add_argument("--output", default="candidates_embedding.pkl")
    ap.add_argument("--merged-output", default="candidates_merged.pkl")
    args = ap.parse_args()

    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())

    print("Loading data...")
    t0 = time.time()
    s1, s2, s3 = load_sources(args.data_dir)
    ground_truth = load_ground_truth(args.data_dir)
    print(f"  S1={len(s1)} S2={len(s2)} S3={len(s3)} ground_truth entries={len(ground_truth)} "
          f"({time.time() - t0:.1f}s)")

    if args.sample_n:
        s1, ground_truth = sample_s1(s1, ground_truth, args.sample_n)
        print(f"\n--sample-n {args.sample_n}: DRY RUN on a subset "
              f"(S2/S3 kept full-size; only S1 + its ground truth is sampled).")
        print(f"  sampled S1={len(s1)} ground_truth entries={len(ground_truth)}")

    print(f"\nLoading embedding model: {args.model_name} (device={args.device or 'auto'})...")
    t0 = time.time()
    model = eb.load_model(args.model_name, device=args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"\nGenerating embedding candidates (fields={fields}, top_k={args.top_k}, "
          f"index_type={args.index_type})...")
    t0 = time.time()
    emb_candidates = eb.generate_embedding_candidates(
        s1, s2, s3, model=model, model_name=args.model_name, fields=fields,
        top_k=args.top_k, batch_size=args.batch_size, index_type=args.index_type,
        cache_dir=args.cache_dir, verbose=True,
    )
    dt = time.time() - t0
    print(f"  {len(emb_candidates)} embedding candidate pairs generated in {dt:.1f}s")

    emb_recall = string_blocking.evaluate_blocking_recall(emb_candidates, ground_truth)
    print(f"  embedding-only candidate recall ceiling: {emb_recall['candidate_recall']:.4f} "
          f"({emb_recall['total_found']}/{emb_recall['total_true_matches']} true matches reachable)")

    emb_candidates.to_pickle(args.output)
    print(f"  saved embedding candidates -> {args.output}")

    if args.existing_candidates:
        print(f"\nMerging with existing candidates from {args.existing_candidates}...")
        existing = pd.read_pickle(args.existing_candidates)
        existing_recall = string_blocking.evaluate_blocking_recall(existing, ground_truth)
        print(f"  existing (string-only) candidate recall: {existing_recall['candidate_recall']:.4f}")

        merged = eb.merge_candidate_sources(existing, emb_candidates, top_k=args.top_k)
        merged_recall = string_blocking.evaluate_blocking_recall(merged, ground_truth)
        print(f"  MERGED candidate recall: {merged_recall['candidate_recall']:.4f} "
              f"({merged_recall['total_found']}/{merged_recall['total_true_matches']}) "
              f"-- uplift vs string-only: "
              f"{merged_recall['candidate_recall'] - existing_recall['candidate_recall']:+.4f}")
        print(f"  merged candidate pairs: {len(merged)}")
        merged.to_pickle(args.merged_output)
        print(f"  saved merged candidates -> {args.merged_output}")
    else:
        print("\n(no --existing-candidates given, so only embedding-alone recall was measured. "
              "Pass your string-blocking candidates_df pickle to see the number that actually "
              "matters: MERGED recall, since that's what you'd feed the classifier.)")


if __name__ == "__main__":
    main()
