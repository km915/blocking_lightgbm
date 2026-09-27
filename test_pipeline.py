"""Synthetic smoke test: fabricate tiny S1/S2/S3 tables with known matches,
run blocking + features + metrics end-to-end, and check nothing crashes and
recall/scoring behave sanely. This does NOT replace testing on the real data
(column values, typo patterns, and scale all differ) but it catches wiring
bugs for free before you burn a submission on them.
"""
import pandas as pd
from src.blocking import generate_candidates, evaluate_blocking_recall, check_cross_country_matches, prepare_frame
from src.features import pair_features
from src.metrics import macro_f_beta, parse_id_list

s1 = pd.DataFrame([
    {"entity_id": "S1-001", "business_name": "Acme Corporation", "business_address": "123 Main St, Springfield", "country": "US"},
    {"entity_id": "S1-002", "business_name": "Globex Pvt Ltd", "business_address": "45 Nehru Road, Mumbai 400001", "country": "India"},
    {"entity_id": "S1-003", "business_name": "Standalone Widgets Inc", "business_address": "9 Lonely Ave", "country": "US"},
    {"entity_id": "S1-004", "business_name": "Cafe de Paris SARL", "business_address": "10 Rue de Rivoli, Paris 75001", "country": "France"},
])

s2 = pd.DataFrame([
    {"entity_id": "S2-001", "business_name": "Acme Corp", "business_address": "123 Main Street, Springfield", "country": "US"},
    {"entity_id": "S2-002", "business_name": "Totally Unrelated LLC", "business_address": "1 Nowhere Rd", "country": "US"},
    {"entity_id": "S2-003", "business_name": "Globex Private Limited", "business_address": "45 Nehru Rd, Mumbai", "country": "India"},
])

s3 = pd.DataFrame([
    {"entity_id": "S3-001", "business_name": "Acme Corporation", "business_address": "123 Main St., Springfield", "country": "US"},
    {"entity_id": "S3-002", "business_name": "Cafe de Paris", "business_address": "10 Rue de Rivoli, 75001 Paris", "country": "France"},
])

# Ground truth: S1-001 matches S2-001 and S3-001; S1-002 matches S2-003;
# S1-003 is a true singleton; S1-004 matches S3-002.
ground_truth = {
    "S1-001": {"S2-001", "S3-001"},
    "S1-002": {"S2-003"},
    "S1-003": set(),
    "S1-004": {"S3-002"},
}

print("=== country-partition sanity check ===")
cross = check_cross_country_matches(s1, s2, s3, ground_truth)
print(f"cross-country ground-truth matches: {cross} (expect 0 for this toy set)")

print("\n=== blocking ===")
cands = generate_candidates(s1, s2, s3, top_k=5, same_country_only=True)
print(cands.sort_values(["source1_entity_id", "blocking_score"], ascending=[True, False]))

recall_stats = evaluate_blocking_recall(cands, ground_truth)
print("\nrecall stats:", recall_stats)
assert recall_stats["candidate_recall"] == 1.0, "blocking should have caught every toy match"

print("\n=== features ===")
s1p = prepare_frame(s1).set_index("entity_id")
s2p = prepare_frame(s2).set_index("entity_id")
s3p = prepare_frame(s3).set_index("entity_id")
ref_lookup = pd.concat([s2p, s3p])

rows = []
for _, r in cands.iterrows():
    a = s1p.loc[r["source1_entity_id"]]
    b = ref_lookup.loc[r["candidate_entity_id"]]
    feats = pair_features(
        a["name_norm"], b["name_norm"], a["name_core"], b["name_core"],
        a["addr_norm"], b["addr_norm"], a["country"], b["country"],
        a["postal"], b["postal"], r["blocking_score"],
    )
    feats["source1_entity_id"] = r["source1_entity_id"]
    feats["candidate_entity_id"] = r["candidate_entity_id"]
    feats["label"] = int(r["candidate_entity_id"] in ground_truth.get(r["source1_entity_id"], set()))
    rows.append(feats)

feat_df = pd.DataFrame(rows)
print(feat_df[["source1_entity_id", "candidate_entity_id", "name_levenshtein",
               "addr_levenshtein", "postal_match", "label"]])

print("\n=== a trivial 'classifier' (threshold on name_levenshtein) to prove the metric wiring ===")
threshold = 0.7
preds = {s1_id: set() for s1_id in s1["entity_id"]}
for _, r in feat_df.iterrows():
    if r["name_levenshtein"] >= threshold:
        preds[r["source1_entity_id"]].add(r["candidate_entity_id"])

score = macro_f_beta(preds, ground_truth)
print("predictions:", preds)
print("macro F_0.5:", round(score, 4))
assert score > 0.9, "toy example should score very high with an obvious threshold"

print("\nALL SMOKE TESTS PASSED")
