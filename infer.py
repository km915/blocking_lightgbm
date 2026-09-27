"""
Generate matching_results.tsv and candidate_pairs.tsv for the test set.

Usage:
    python3 infer.py --data-dir dataset/test --artifacts-dir artifacts --out-dir output

Important: candidate_pairs.tsv must be the exact candidate set the model
scored (not an earlier, unfiltered blocking pass), and every ID in
matching_results.tsv must appear in candidate_pairs.tsv. This script writes
both from the same in-memory candidate set, so that invariant holds
automatically -- don't refactor it so they diverge.

Uses the same chunked, ID-coded feature building as train.py (see
src/features.py's IdCodec / build_features_chunked) -- entity ID strings are
only materialized right before writing the output files, not carried through
scoring.
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.blocking import generate_candidates, prepare_frame
from src.features import build_features_chunked, FEATURE_NAMES


def _codes_to_joined_strings(entity_codes, item_codes, codec) -> dict:
    """Group item_codes by entity_codes, decode to real ID strings, sort,
    comma-join. Returns {entity_code: 'ID1,ID2,...'}."""
    if len(entity_codes) == 0:
        return {}
    df = pd.DataFrame({"e": entity_codes, "i": item_codes})
    grouped = df.groupby("e")["i"].apply(lambda codes: sorted(set(codes)))
    out = {}
    for e, code_list in grouped.items():
        ids = codec.decode(np.array(code_list, dtype=np.int32))
        out[int(e)] = ",".join(sorted(ids.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/test")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--top-k", type=int, default=None, help="override top_k from training config")
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold")
    ap.add_argument("--chunk-size", type=int, default=2_000_000,
                     help="Feature-building chunk size -- lower this if you still run out of RAM.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.artifacts_dir, "config.json")) as f:
        config = json.load(f)
    model = joblib.load(os.path.join(args.artifacts_dir, "lgbm_model.pkl"))

    top_k = args.top_k or config["top_k"]
    threshold = args.threshold if args.threshold is not None else config["threshold"]
    same_country_only = config["same_country_only"]
    print(f"Using top_k={top_k} threshold={threshold} same_country_only={same_country_only}")

    print("Loading test data...")
    s1 = pd.read_csv(os.path.join(args.data_dir, "test_source1.tsv"), sep="\t", dtype=str)
    s2 = pd.read_csv(os.path.join(args.data_dir, "test_source2.tsv"), sep="\t", dtype=str)
    s3 = pd.read_csv(os.path.join(args.data_dir, "test_source3.tsv"), sep="\t", dtype=str)
    print(f"  S1={len(s1)} S2={len(s2)} S3={len(s3)}")
    print(f"  test countries seen: {sorted(s1['country'].dropna().unique())}")

    print("Generating candidates...")
    candidates = generate_candidates(s1, s2, s3, top_k=top_k, same_country_only=same_country_only)
    print(f"  {len(candidates)} candidate pairs")

    print("Building features (chunked, ID-coded) + scoring...")
    s1p = prepare_frame(s1)
    ref_lookup = pd.concat([prepare_frame(s2), prepare_frame(s3)], ignore_index=True)
    X, s1_codes, cand_codes, _, s1_codec, ref_codec = build_features_chunked(
        candidates, s1p, ref_lookup, ground_truth=None, chunk_size=args.chunk_size)

    if len(X):
        prob = model.predict(X)
    else:
        prob = np.array([])

    print("Writing candidate_pairs.tsv (the exact set just scored)...")
    cand_out = _codes_to_joined_strings(s1_codes, cand_codes, ref_codec)
    all_s1_ids = list(s1["entity_id"])
    all_s1_codes = s1_codec.codes(np.array(all_s1_ids))  # vectorized, matches build order
    with open(os.path.join(args.out_dir, "candidate_pairs.tsv"), "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, code in zip(all_s1_ids, all_s1_codes):
            f.write(f"{s1_id}\t{cand_out.get(int(code), '') if code >= 0 else ''}\n")

    print(f"Writing matching_results.tsv (threshold={threshold})...")
    if len(prob):
        keep_mask = prob >= threshold
        match_out = _codes_to_joined_strings(s1_codes[keep_mask], cand_codes[keep_mask], ref_codec)
    else:
        match_out = {}
    n_with_matches = sum(1 for code in all_s1_codes if match_out.get(int(code), ""))
    with open(os.path.join(args.out_dir, "matching_results.tsv"), "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, code in zip(all_s1_ids, all_s1_codes):
            f.write(f"{s1_id}\t{match_out.get(int(code), '') if code >= 0 else ''}\n")

    print(f"  {n_with_matches}/{len(all_s1_ids)} S1 entities predicted to have >=1 match "
          f"({len(all_s1_ids) - n_with_matches} predicted singletons)")
    print(f"\nDone. Files written to {args.out_dir}/")
    print("Now run utils/validate_submission.py against these before uploading.")


if __name__ == "__main__":
    main()
