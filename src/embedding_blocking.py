"""
Embedding-based semantic blocking for the entity resolution challenge.

WHY THIS EXISTS: the string-similarity blocker (blocking.py) plateaus around
78-80% candidate recall. That signal is fundamentally character-overlap
based (exact keys, shared tokens, sorted-neighborhood on normalized text) --
it structurally cannot see two strings as related if they don't share
characters/tokens, no matter how you tune top_k or window size. Cases that
look nothing alike at the character level but ARE the same business:
  - heavy abbreviation / DBA-style renames ("International Business
    Machines" vs "IBM Global Services")
  - cross-script name pairs where transliteration doesn't land on the exact
    English spelling ("urban" vs ITRANS "arvana")
  - paraphrased addresses (landmark references, reordered/renamed components)
are invisible to character-overlap blocking by construction, not because of
a tuning gap. A multilingual sentence embedding model is trained so that
semantically-equivalent text -- across languages/scripts, with different
surface forms -- lands close together in vector space, which is exactly the
complementary signal needed here.

ARCHITECTURE:
  1. Encode business_name (and optionally business_address) for every S1
     entity and every S2/S3 record into dense vectors with a small
     multilingual sentence-embedding model.
  2. Partition by country (same reasoning as blocking.py: never a different
     code path per country, just a way to shrink the search space).
  3. Build an ANN index (FAISS) per (country, source) partition over the
     S2/S3 vectors, and retrieve the top_k nearest S1 queries against it.
     This is what keeps this O(n log n)-ish instead of all-pairs -- at S2~5M/
     S3~5.3M records, brute-force cosine similarity against ~2.2M S1 queries
     is ~10^13 comparisons and is not happening.
  4. Output candidate pairs in the SAME schema as blocking.generate_candidates
     (source1_entity_id, candidate_entity_id, source, blocking_score) so this
     can be unioned with the string-similarity candidates via
     merge_candidate_sources() below, or fed to the classifier as an
     additional row source before feature building.

MODEL CHOICE (both MIT/Apache-2.0, both far under the 8B-parameter cap):
  - intfloat/multilingual-e5-small (MIT, 118M params, 384-dim, 94 languages).
    Fast. Needs "query: " / "passage: " prefixes on the input text or you
    lose meaningful quality -- this module adds them for you.
  - sentence-transformers/LaBSE (Apache-2.0, ~471M params, 768-dim,
    109 languages). Purpose-built for cross-lingual bitext retrieval, which
    is closer to this exact problem (aligning translations/transliterations
    of the same underlying text) than e5's general retrieval training. No
    prefix needed. ~4x the params of e5-small -> proportionally slower to
    encode.
  Given the challenge window closes today, DEFAULT IS e5-small. Swap to
  LaBSE (pass model_name="sentence-transformers/LaBSE", it has no prefix
  requirement so pass query_prefix="", passage_prefix="") only if you have
  encoding time to spare and want to squeeze more cross-script recall.

REAL RISKS, stated plainly:
  - Encoding ~12.5M short strings (S1+S2+S3 name field) is the dominant
    cost, not the FAISS search. On a single GPU (even a modest one) this is
    on the order of 30-90 minutes for e5-small. On CPU only, expect it to be
    10-20x slower -- potentially many hours, which may not fit your
    remaining window. If you have no GPU: run the --sample-n dry run first,
    consider a free Colab GPU for the encode step only (FAISS indexing/
    search itself is cheap and fine on CPU), and encode name-only before
    ever attempting to also embed address.
  - Memory: vectors are only ever held for one (country, source) partition
    at a time. The largest partition here is the US split of S2/S3, at
    roughly 3-3.5M records x 384 floats x 4 bytes =~ 5GB for e5-small,
    doubled transiently during FAISS index add(). LaBSE (768-dim) doubles
    that again. This is why partitioning by country isn't just a speed
    optimization here, it's what keeps this fitting in memory at all.
  - Embedding models can and do produce high cosine similarity for
    different-but-related businesses (e.g. two unrelated franchises of the
    same chain, "Subway" #1 vs "Subway" #2 in different addresses). This is
    a precision risk that string blocking's exact-key signal doesn't have.
    Given F_0.5 punishes false merges 2x harder than misses, treat
    blocking_score from this module as ONE feature into the classifier
    (features.py), not a standalone acceptance threshold -- let the
    classifier learn how much to trust it relative to postal_match,
    country_match, name_char3_jaccard, etc.
  - This does NOT branch on country value anywhere except to partition the
    search space, same invariant as blocking.py, so France should behave
    like any other value seen or unseen in training.

WHAT THIS MODULE DELIBERATELY DOES NOT DO:
  - No re-implementation of your feature engineering / classifier. This only
    produces (and optionally scores) candidate pairs, exactly like
    blocking.py's generate_candidates -- so it plugs into your existing
    pipeline at the same point.
  - No external API/database lookups. The embedding model is a local,
    pretrained-weights-only checkpoint (downloaded once, cached locally, run
    entirely offline afterward) -- it never looks up business identities
    externally, so it doesn't touch the "no external lookup" rule. It DOES
    require one-time internet access to download the model weights from
    Hugging Face; do that download before you're offline / rate-limited.
"""
import os
import re
import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

_PAIR_COLS = ["source1_entity_id", "candidate_entity_id", "source", "blocking_score"]

# Models known to need asymmetric "query: " / "passage: " prefixes to hit
# their trained performance (the E5 family). Anything not in this set is
# assumed to be prefix-free (e.g. LaBSE, MiniLM/mpnet paraphrase models).
_E5_PREFIX_MODELS = {
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/e5-small", "intfloat/e5-base", "intfloat/e5-large",
}


def default_prefixes(model_name: str):
    """Returns (query_prefix, passage_prefix) for a known model family."""
    if model_name in _E5_PREFIX_MODELS:
        return "query: ", "passage: "
    return "", ""


# ---------------------------------------------------------------------------
# Model loading / encoding
# ---------------------------------------------------------------------------

def load_model(model_name: str = "intfloat/multilingual-e5-small", device: str = None):
    """Loads a sentence-transformers model. Requires internet access on
    first call to fetch weights from Hugging Face (cached locally after
    that) -- this is a model-weights download, not a business-identity
    lookup, so it doesn't touch the "no external lookup" rule, but do it
    while you still have connectivity."""
    from sentence_transformers import SentenceTransformer
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model


def embed_texts(model, texts, batch_size: int = 512, prefix: str = "",
                 show_progress: bool = False) -> np.ndarray:
    """Encodes a list of strings to L2-normalized float32 vectors (so inner
    product == cosine similarity, which is what the FAISS indices below
    assume)."""
    if prefix:
        texts = [prefix + t for t in texts]
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Text field construction
# ---------------------------------------------------------------------------

def prepare_text(df: pd.DataFrame, field: str = "name") -> pd.DataFrame:
    """Builds the `_emb_text` column to encode. Deliberately uses the RAW
    business_name/business_address, not the ITRANS-transliterated or
    ASCII-folded output of normalize.py: multilingual embedding models are
    trained on natural text in each script, so feeding them pseudo-English
    ITRANS output is out-of-distribution for the model and likely to hurt
    more than help. Let the embedding model do its own cross-script
    alignment; save normalize.py's transliteration for the string-similarity
    signals where it's the only option."""
    df = df.copy()
    name = df["business_name"].fillna("").astype(str).str.strip()
    addr = df["business_address"].fillna("").astype(str).str.strip()
    if field == "name":
        df["_emb_text"] = name
    elif field == "address":
        df["_emb_text"] = addr
    elif field == "name_address":
        df["_emb_text"] = np.where(addr != "", name + " -- " + addr, name)
    else:
        raise ValueError(f"unknown field: {field!r}")
    df["country"] = df["country"].fillna("")
    return df


# ---------------------------------------------------------------------------
# FAISS index construction
# ---------------------------------------------------------------------------

def build_faiss_index(vectors: np.ndarray, index_type: str = "auto"):
    """Builds a cosine-similarity ANN index (IndexFlatIP on normalized
    vectors == cosine). Index type auto-scales with partition size so a
    ~10k-record partition doesn't pay IVF training overhead, and a
    ~3M-record partition (the largest expected here, after country
    partitioning) doesn't try to build an exact O(n) per-query index."""
    if faiss is None:
        raise ImportError(
            "faiss is required: pip install faiss-cpu --break-system-packages "
            "(faiss-gpu is not needed -- indexing/search is cheap here even on "
            "CPU; only the sentence embedding step benefits from a GPU)")
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, d = vectors.shape

    if index_type == "auto":
        if n <= 20_000:
            index_type = "flat"
        elif n <= 500_000:
            index_type = "ivfflat"
        else:
            index_type = "ivfpq"

    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
        return index

    nlist = max(1, min(4096, int(4 * np.sqrt(n))))
    quantizer = faiss.IndexFlatIP(d)

    if index_type == "ivfflat":
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    elif index_type == "ivfpq":
        m = 48
        while d % m != 0 and m > 1:
            m -= 1
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(f"unknown index_type: {index_type!r}")

    train_size = min(n, max(nlist * 40, 100_000))
    if train_size < n:
        rng = np.random.default_rng(0)
        train_idx = rng.choice(n, size=train_size, replace=False)
        index.train(vectors[train_idx])
    else:
        index.train(vectors)
    index.add(vectors)
    index.nprobe = min(nlist, 32)
    return index


# ---------------------------------------------------------------------------
# Per-partition embedding cache (crash-resumable at partition granularity)
# ---------------------------------------------------------------------------

def _cache_file(cache_dir, source_name, country, field):
    safe_country = re.sub(r"[^A-Za-z0-9]+", "_", str(country)) or "UNK"
    return os.path.join(cache_dir, f"emb_{source_name}_{safe_country}_{field}.npz")


def get_or_compute_embeddings(model, sub_df, id_col, text_col, prefix, cache_dir,
                               source_name, country, field, batch_size=512,
                               verbose=True) -> np.ndarray:
    """Computes embeddings for one (source, country, field) partition, or
    loads them from disk if a cache file already covers exactly this set of
    ids. Caching is at whole-partition granularity, not per-row: simpler and
    good enough given the challenge timeline, and it still means a crash
    while processing, say, the US/S2 partition doesn't cost you the India/S2
    partition you already finished."""
    path = _cache_file(cache_dir, source_name, country, field) if cache_dir else None
    ids = sub_df[id_col].to_numpy()

    if path and os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        cached_ids = data["ids"]
        if len(cached_ids) == len(ids) and set(cached_ids.tolist()) == set(ids.tolist()):
            if verbose:
                print(f"    [cache hit] {os.path.basename(path)}")
            order = {eid: i for i, eid in enumerate(cached_ids)}
            reorder = np.array([order[eid] for eid in ids])
            return data["vecs"][reorder]
        elif verbose:
            print(f"    [cache stale, recomputing] {os.path.basename(path)}")

    texts = sub_df[text_col].tolist()
    vecs = embed_texts(model, texts, batch_size=batch_size, prefix=prefix)

    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, ids=ids, vecs=vecs)
    return vecs


# ---------------------------------------------------------------------------
# Candidate pair capping / merging (schema-compatible with blocking.py)
# ---------------------------------------------------------------------------

def cap_top_k(pairs: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Dedupes (source1_entity_id, candidate_entity_id) by max score, keeps
    the top_k highest-scoring candidates per S1 entity, and (re)derives the
    `source` column from the candidate id's prefix, so this works whether
    `pairs` came from one embedding field, several unioned together, or a
    union with blocking.py's own output."""
    if len(pairs) == 0:
        return pd.DataFrame(columns=_PAIR_COLS)
    pairs = pairs.groupby(["source1_entity_id", "candidate_entity_id"],
                           as_index=False)["blocking_score"].max()
    pairs = pairs.sort_values(["source1_entity_id", "blocking_score"], ascending=[True, False])
    pairs["_rank"] = pairs.groupby("source1_entity_id").cumcount()
    pairs = pairs[pairs["_rank"] < top_k].drop(columns="_rank")
    pairs["source"] = pairs["candidate_entity_id"].str.slice(0, 2)
    return pairs[_PAIR_COLS].reset_index(drop=True)


def merge_candidate_sources(*candidate_dfs, top_k: int = 30) -> pd.DataFrame:
    """Unions candidate pairs from multiple blocking passes (e.g.
    blocking.generate_candidates()'s string-similarity output and this
    module's generate_embedding_candidates() output), keeping the max score
    per (s1, candidate) pair across sources, then re-applies the top_k cap so
    the merged set stays bounded (this is the recommended way to plug
    embedding blocking in as an ADDITIONAL signal per the challenge doc's
    option 4a)."""
    cols = ["source1_entity_id", "candidate_entity_id", "blocking_score"]
    combined = pd.concat([df[cols] for df in candidate_dfs if len(df)], ignore_index=True)
    return cap_top_k(combined, top_k)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_embedding_candidates(s1_df, s2_df, s3_df, model=None,
                                   model_name: str = "intfloat/multilingual-e5-small",
                                   fields=("name",), top_k: int = 30,
                                   same_country_only: bool = True, batch_size: int = 512,
                                   index_type: str = "auto", cache_dir: str = None,
                                   query_prefix: str = None, passage_prefix: str = None,
                                   verbose: bool = True) -> pd.DataFrame:
    """Returns a DataFrame with the SAME schema as
    blocking.generate_candidates: source1_entity_id, candidate_entity_id,
    source, blocking_score (blocking_score here is cosine similarity in
    [-1, 1], typically [0, 1] in practice for this kind of text).

    fields: which text field(s) to embed and retrieve on. Defaults to
    ("name",) -- the highest-value, lowest-cost signal, and the one your
    parallel string-blocking attempt already flagged as the pain point
    (cross-script names). Add "address" only if you have encoding time left
    after validating name-only recall uplift; it roughly doubles encode time.

    Mirrors blocking.py's partition-by-country loop and top_k capping so the
    two are drop-in comparable / mergeable via merge_candidate_sources().
    """
    if model is None:
        model = load_model(model_name)
    if query_prefix is None or passage_prefix is None:
        qp, pp = default_prefixes(model_name)
        query_prefix = qp if query_prefix is None else query_prefix
        passage_prefix = pp if passage_prefix is None else passage_prefix

    records = []
    for field in fields:
        s1p = prepare_text(s1_df, field)
        for src_name, src_df in (("S2", s2_df), ("S3", s3_df)):
            if src_df is None or len(src_df) == 0:
                continue
            srcp = prepare_text(src_df, field)

            groups = sorted(set(s1p["country"]) | set(srcp["country"])) if same_country_only else [None]
            for country in groups:
                if same_country_only:
                    s1_sub = s1p[s1p["country"] == country]
                    ref_sub = srcp[srcp["country"] == country]
                else:
                    s1_sub, ref_sub = s1p, srcp
                if len(s1_sub) == 0 or len(ref_sub) == 0:
                    continue
                if verbose:
                    print(f"    [emb:{field}] [{src_name}] country={country!r}: "
                          f"{len(s1_sub)} queries x {len(ref_sub)} reference records...")

                ref_vecs = get_or_compute_embeddings(
                    model, ref_sub, "entity_id", "_emb_text", passage_prefix,
                    cache_dir, src_name, country, field, batch_size, verbose)
                q_vecs = get_or_compute_embeddings(
                    model, s1_sub, "entity_id", "_emb_text", query_prefix,
                    cache_dir, f"S1for{src_name}", country, field, batch_size, verbose)

                index = build_faiss_index(ref_vecs, index_type=index_type)
                k = min(top_k, len(ref_sub))
                sims, idxs = index.search(q_vecs, k)

                ref_ids = ref_sub["entity_id"].to_numpy()
                s1_ids = s1_sub["entity_id"].to_numpy()
                flat_idx = idxs.reshape(-1)
                valid = flat_idx >= 0
                s1_rep = np.repeat(s1_ids, k)

                part = pd.DataFrame({
                    "source1_entity_id": s1_rep[valid],
                    "candidate_entity_id": ref_ids[flat_idx[valid]],
                    "blocking_score": sims.reshape(-1)[valid],
                })
                records.append(part)

    if not records:
        return pd.DataFrame(columns=_PAIR_COLS)
    combined = pd.concat(records, ignore_index=True)
    return cap_top_k(combined, top_k)
