# Amazon ML Challenge 2026 — Business Entity Resolution — Starter Kit

**First thing: check your actual clock.** Window is 25 Sept 00:00 IST → 27 Sept
23:59 IST. You are already on day 2. Budget your remaining hours before doing
anything else — this plan assumes well under 36 hours are left, not 72.

This kit is a *tested scaffold*, not a finished solution — I built and ran it
against synthetic data (see `test_pipeline.py`) to verify the wiring (blocking
→ features → scoring) is correct, including the exact F_0.5 formula from the
problem statement. It has **never seen your real data**, so treat the first
hour as "get this running on the real files and see what breaks."

## ⚠️ Scale note (read this before rerunning)

Your real data (S1≈2.2M, S2≈5M, S3≈5.3M) is **far larger** than what the
original blocking implementation could handle — it used a brute-force
nearest-neighbor search that's effectively all-pairs comparison, fine for
thousands of rows, infeasible for millions. It has been rewritten to use
vectorized hash-joins and a sorted-neighborhood scan instead (see
`src/blocking.py`'s docstring). The feature-building step was also rewritten
— the original per-row `.loc` lookups benchmarked at >100x slower than a
single vectorized merge at this scale.

**Realistic timing at your actual scale**, extrapolated from benchmarks on
synthetic data at 1/10th–1/100th scale on ordinary hardware:
- Data loading: ~1-3 min
- Blocking: ~5-10 min
- Feature building: ~20-30 min
- LightGBM training (5-fold): untested at full scale — watch the per-fold
  progress prints; if it's taking a very long time, see "If it's still too
  slow" below.

**Before committing to a full run, dry-run it:**
```bash
python3 train.py --data-dir dataset/train --top-k 20 --sample-n 20000
```
This samples S1 down to 20k entities (keeping S2/S3 full-size, so blocking
behaves realistically) and should finish in roughly a minute or two —
enough to catch config/path/environment issues without waiting 30+ minutes.
Scale `--sample-n` up (50k, 200k...) to get a feel for how time grows before
running the full 2.2M.

**If it's still too slow at full scale:**
- Lower `--top-k` (fewer candidates per entity = less feature-building work,
  at some cost to recall — check the recall number before and after).
- In `src/blocking.py`, lower `window` in the sorted-neighborhood calls.
- Downsample negative pairs before training (keep all positives, randomly
  sample e.g. 20-30x as many negatives) — this cuts LightGBM training set
  size substantially with little information loss, since the vast majority
  of negatives are unambiguous non-matches.

## Memory fix (this version: features.py, train.py, infer.py)

`train.py`/`infer.py` now build features via `build_features_chunked` in
`src/features.py` instead of one big merge:
- Entity ID strings are converted to compact int32 codes internally (see
  `IdCodec`) — a real ID costs ~70-90 bytes as a repeated Python string but
  4 bytes as a code, and this repeats once per candidate row.
- Candidates are processed in `--chunk-size` row batches (default 2,000,000)
  so peak memory doesn't grow with total candidate count.

Measured (not estimated) on synthetic data: the old single-merge approach
was OOM-killed at 2M candidate rows in a 3.9GB test environment; the new
approach handled 2M rows in 2.54GB and 5M rows in 2.61GB (confirming memory
stays flat as row count grows, dominated by fixed overhead not candidate
count). At full scale, the feature matrix itself (float32, unavoidable) is
the main cost — roughly `rows × 19 × 4 bytes`. If you still run out of RAM,
lower `--chunk-size` first (e.g. 500000) before lowering `--top-k`.

## Your real recall number and what it means

Your first real run showed **68% candidate recall** with the old parameters
— well below the 95% target. I tried to reproduce this with synthetic data
at a similar scale and, with the same old parameters, got **98.8% recall** —
meaning your real data's noise pattern is NOT primarily the character-level
typos I'd simulated. That's expected: the problem statement calls out DBA
names, heavy abbreviations, and missing address components, which behave
very differently from random typos and need different signals.

**Two things changed as a result:**

1. **`src/blocking.py` now has more signal types**, all still fast/vectorized:
   - Token-based blocking (shared significant *words*, robust to a typo in
     one word since other words in a multi-word name still match — doesn't
     degrade with reference-pool size the way sorted-neighborhood does)
   - A second sorted-neighborhood pass on the *reversed* name string (catches
     typos near the start of a name, which the forward pass misses)
   - Bumped default `window` (8→40) and `max_group_size` (300→2000), since at
     millions-of-rows scale the old values were too tight
2. **`diagnose_missed_matches.py` — run this first, on your real data,
   before assuming any fix worked:**
   ```bash
   python3 diagnose_missed_matches.py --data-dir dataset/train --sample-n 20000 --top-k 20 --show 15
   ```
   This prints actual (S1 record, missed true match) pairs side by side.
   **Look at them.** If names share no text overlap but postal/address align,
   your data has real DBA-type matches — no amount of string-similarity
   tuning fixes that, and the next lever is embedding-based semantic
   blocking (see "Stretch goals"), which should become a priority, not a
   stretch, if this is what you see. If the names DO look similar but still
   weren't caught, paste me 5-10 examples and I'll tune the actual signals
   to what's really happening instead of guessing again.



Read `train.py`'s printed `candidate recall ceiling` the moment it finishes.
That is the maximum possible score your classifier could ever reach — if
blocking only proposes 80% of true matches as candidates, your F_0.5 ceiling
is roughly 0.8 no matter how good the classifier is afterward (precision-heavy
metric aside). **Spend your first real hours on blocking recall, not on the
classifier.** This is the opposite of where instinct usually pulls people.

## Architecture

```
train_source1/2/3.tsv ──┐
                         ├─→ blocking.py (candidate generation)
train_ground_truth.tsv ─┘         │
                                   ▼
                    candidate pairs (source1_id, candidate_id, source, blocking_score)
                                   │
                                   ▼
                    features.py (pairwise similarity features)
                                   │
                                   ▼
                    train.py: LightGBM + GroupKFold + threshold tuning
                       against the EXACT competition metric (metrics.py)
                                   │
                                   ▼
                    infer.py: score test candidates, apply threshold,
                       write matching_results.tsv + candidate_pairs.tsv
```

## Run it (in priority order)

```bash
pip install -r requirements.txt

# 1. Sanity check the wiring still works on your machine
python3 test_pipeline.py

# 2. Point at the REAL data and look hard at the printed stats
python3 train.py --data-dir dataset/train --top-k 20

# 3. Generate a submission
python3 infer.py --data-dir dataset/test --artifacts-dir artifacts --out-dir output

# 4. Validate locally before burning a submission
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

**The moment you have real data**, before trusting anything above, run these
checks (add a scratch notebook/script — not provided, because I don't know
your data's actual shape):
- Row counts of each file, and how many S1 entities have 0 / 1 / many matches.
- `check_cross_country_matches()` in `src/blocking.py` on the training ground
  truth — confirms whether partitioning candidates by country is safe (it
  almost certainly is, but verify, don't assume).
- Eyeball 20-30 real `business_name` / `business_address` values. Your
  `LEGAL_SUFFIXES` / `STREET_ABBR` dictionaries in `src/normalize.py` are a
  starting guess — extend them with whatever abbreviations you actually see,
  especially for France, which isn't in training data so you'll only see it
  in test — build your normalization to be generic (character/token
  similarity) rather than dictionary-dependent for that reason.
- Dataset scale. If S2/S3 are in the hundreds of thousands per country, the
  TF-IDF nearest-neighbor step in `blocking.py` (brute-force cosine) may be
  slow — see "Scaling blocking" below.

## Why this architecture, and what to try next (in order of expected payoff)

1. **Blocking recall first.** If recall ceiling < ~95%, before touching the
   model:
   - Raise `--top-k`.
   - Add a second blocking signal: `normalize_name_core` phonetic key
     (Soundex/double-metaphone via the `jellyfish` package) catches
     phonetically-similar typos that char n-grams sometimes miss.
   - Add an **embedding-based** blocking pass (see "Stretch goals" below) —
     this is what catches translated/reworded names that string similarity
     can't, which matters given the France/multilingual angle.

2. **Feature richness.** The features in `src/features.py` are string-only.
   High-value additions once you've looked at real data:
   - Parsed address components (city/state/postal) compared separately
     rather than as one blob, if the address format is parseable.
   - Token-level abbreviation-aware comparison (e.g. compare
     `name_core` — suffix-stripped — in addition to full name).
   - Candidate's *rank within its own S1 entity's candidate list* — a
     candidate that's the clear best match for its entity is more likely
     correct than one that's a middling 3rd-best.

3. **Model diversity → ensemble.** Add XGBoost and CatBoost (same feature
   table, `requirements-heavy.txt`) and average probabilities, or stack with
   a logistic regression meta-model on the three OOF probability columns.
   Gradient boosting ensembles are usually where the bulk of the score comes
   from in tasks like this — prioritize this over exotic models.

4. **Threshold policy, not just accuracy.** F_0.5 weights precision 2x, and a
   false merge on a true singleton scores 0.0 for that entity (a full point
   lost), while a missed match on a non-singleton also scores 0.0 for that
   entity if the *only* candidate is missed. In practice this means:
   - The tuned threshold from `train.py` is a good start, but also try a
     **margin rule**: only keep a candidate if its probability clears the
     threshold *and* beats the next-best candidate for that entity by some
     margin (reduces false merges from two similarly-scored candidates).
     This is not implemented yet — worth an experiment if you have time.
   - Don't be tempted to always force at least one match per entity —
     correctly predicting empty is worth a full 1.0.

## Stretch goals (only after a solid GBM baseline is submitted and locked in)

- **Semantic/embedding blocking + features**: use a small multilingual
  sentence-transformer (e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
  MIT licensed, ~120M params — comfortably under the 8B/license constraint)
  to embed names, then add cosine similarity as a feature and/or an ANN
  index (`faiss-cpu` or `sklearn.neighbors`) as an extra blocking signal.
  This is the best lever for the "France isn't in training" difficulty,
  since it generalizes better than string edit-distance to paraphrased
  business names.
- **Cross-encoder fine-tuning**: fine-tune a small encoder (e.g.
  `microsoft/deberta-v3-small`, MIT licensed) on serialized pairs
  (`"[NAME] Acme Corp [ADDR] 123 Main St [SEP] [NAME] Acme Corporation [ADDR] ..."`)
  with a binary head. This is the "Ditto"-style approach that tends to beat
  pure feature-based GBMs on messy text matching, at the cost of needing a
  GPU and more engineering time. Only attempt this if you have a working GBM
  submission already in and time to spare — it's higher effort, higher
  ceiling, higher risk of not finishing.
- Blend GBM ensemble probability with cross-encoder probability (simple
  average or a small stacking model) for the final score.

## Scaling blocking if the real data is large

`_tfidf_topk` in `src/blocking.py` uses `NearestNeighbors(algorithm="brute")`,
which is fine up to tens of thousands of records per country but will get
slow beyond that. If your per-country groups are large:
- Increase `min_df` in the `TfidfVectorizer` call to shrink the vocabulary.
- Swap to an approximate method: `faiss.IndexFlatIP` on L2-normalized TF-IDF
  or embedding vectors, or `datasketch` MinHashLSH on character shingles.
- Process country groups in parallel (they're already independent) —
  straightforward with `multiprocessing` or by just running on a
  many-vCPU AWS box (see `AWS_SETUP.md`).

## Submission discipline (max 5/day)

- Don't submit every local improvement — validate locally with `train.py`'s
  OOF macro F_0.5 first, which is computed with the *exact same function*
  the leaderboard conceptually uses (`src/metrics.py`), so it should track
  the real score closely once your candidate recall is high.
- Reserve at least 1 submission per remaining day as a "safety" — submit
  your current best early, then only replace it with something that beat
  your OOF score locally.
- Keep every submitted `matching_results.tsv` + the model/config that
  produced it (the rules require you to show version history).

## Before you upload anything

1. Run `utils/validate_submission.py` — it catches format issues without
   costing you a submission.
2. Double check: every test S1 entity appears exactly once in both output
   files; `matching_results.tsv` matches are a subset of
   `candidate_pairs.tsv`; no self-matches to Source 1; no duplicate IDs
   within a list.
3. Fill in `Documentation_template.md` as you go, not at the last hour —
   it's required for the final package and (per the guidelines) shortlisting
   considers it.

## Rules to not accidentally break

- No external lookups: no geocoding APIs, no business registry APIs, no
  scraping. Using a generic pretrained open-weight embedding/language model
  (not entity-specific, not an external database) is fine and is the
  intended use of "8B params, MIT/Apache license" — that constraint is about
  *which pretrained checkpoints you may use*, not a ban on pretrained models
  altogether.
- Read files with `sep="\t"` always — they're TSVs specifically because
  addresses and ID lists contain commas; a plain `read_csv` will silently
  produce garbage.
- Don't branch your code on `if country == "US"` / `"India"` anywhere — the
  test set includes France, which never appears in training. Every function
  in this kit is written to be country-agnostic (country is only ever used
  as an equality-comparison feature or a partition key, never to select
  different logic) — keep it that way in anything you add.
"# blocking_lightgbm" 
"# blocking_lightgbm" 
