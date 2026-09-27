"""
Text normalization for business names and addresses.

Design principle: don't hard-code logic to only US/India. Everything here is
generic string cleanup plus abbreviation dictionaries. Extend the dictionaries
freely (e.g. more French/Indian terms) -- that's safe. Do NOT branch behavior
on the `country` field itself; use country only as a *feature*, never as a
switch that changes which normalization function runs. The test set has
France, which never appears in training, so any country-conditional code path
will silently do the wrong thing on it.

TRANSLITERATION: real data includes business names written entirely in
Indic scripts (Devanagari, Bengali, Gujarati, Tamil, etc.), sometimes mixed
with Latin text in the same field (e.g. a Tamil business name followed by
an English "Private Limited", or an English address with a Devanagari state
name). Every blocking/similarity signal in this pipeline operates on shared
characters, so a Latin-script name and its Indic-script counterpart share
ZERO characters and are otherwise invisible to it -- this isn't a tuning
problem, it's a structural blind spot. `transliterate_mixed_script` converts
Indic-script RUNS within a string to a Latin phonetic approximation (via
indic_transliteration's ITRANS scheme) while leaving Latin/ASCII runs
untouched, so mixed-script strings are handled correctly. It's called from
inside `basic_clean`, so every function in this module gets it for free.
"""
import re
import unicodedata

try:
    from unidecode import unidecode
except ImportError:  # pragma: no cover
    def unidecode(x):
        return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")

try:
    from indic_transliteration import sanscript
    _HAS_INDIC = True
except ImportError:  # pragma: no cover
    _HAS_INDIC = False

# Unicode block ranges for scripts indic_transliteration supports and that
# are plausible in this dataset. Add more here if you spot other scripts
# (Telugu, Kannada, Malayalam are included pre-emptively even without a
# confirmed example, since they're common Indian business-register scripts).
_SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "oriya": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}


def _char_script(ch):
    cp = ord(ch)
    for name, (lo, hi) in _SCRIPT_RANGES.items():
        if lo <= cp <= hi:
            return name
    return None


def transliterate_mixed_script(text: str) -> str:
    """Convert Indic-script runs within `text` to Latin (ITRANS), leaving
    Latin/ASCII/digit/punctuation runs untouched, so a mixed-script string
    like 'அரிஹந்த் Foundation Private Limited' becomes
    'arihandh Foundation Private Limited' rather than being left opaque.
    Cheap no-op for pure-ASCII text (the common case) via the isascii() check."""
    if not text or text.isascii() or not _HAS_INDIC:
        return text

    runs = []
    cur_script, cur_chars = None, []
    for ch in text:
        s = _char_script(ch)
        if s != cur_script:
            if cur_chars:
                runs.append((cur_script, "".join(cur_chars)))
            cur_script, cur_chars = s, [ch]
        else:
            cur_chars.append(ch)
    if cur_chars:
        runs.append((cur_script, "".join(cur_chars)))

    out = []
    for script, chunk in runs:
        if script is None:
            out.append(chunk)
        else:
            try:
                scheme = getattr(sanscript, script.upper())
                t = sanscript.transliterate(chunk, scheme, sanscript.ITRANS)
                # strip ITRANS notation marks that would otherwise wrongly
                # split one word into two once basic_clean's punctuation
                # stripping turns them into spaces
                t = t.replace("~", "").replace("'", "")
                out.append(t)
            except Exception:
                out.append(chunk)  # fail safe: leave the original script in place
    return "".join(out)

# Legal / business-form suffixes -> canonical token. Extend this list as you
# see more variants in the real data (do an EDA pass on the most common last
# tokens of business_name to find what's missing).
LEGAL_SUFFIXES = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc",
    "llp": "llp",
    "pvt": "pvt", "private": "pvt",
    "plc": "plc",
    "gmbh": "gmbh",
    "sa": "sa",
    "sarl": "sarl",
    "spa": "spa",
    "pte": "pte",
    "bv": "bv",
    "nv": "nv",
    "ag": "ag",
    "sas": "sas",
    "eurl": "eurl",
    # Common ITRANS-transliterated phonetic renderings of these same words,
    # seen directly from transliterating real Devanagari/Bengali/Gujarati
    # business names in this dataset ("प्राइवेट"/"প্রাইভেট" -> "private" but
    # transliterates as "praiveta"/"praibheta", etc). The dictionary-based
    # exact-match approach otherwise can't canonicalize these, since it only
    # knows the literal English spelling. Add more as you observe them.
    # NOTE: keys must be lowercase -- basic_clean lowercases before this
    # dict is consulted, so a mixed-case key here would silently never match.
    "praiveta": "pvt", "praibheta": "pvt",
    "limiteda": "ltd",
    "elaelapi": "llp",  # "एलएलपी" = "L-L-P" spelled out letter by letter
}

# Street / address abbreviations (English + a few French ones since the test
# set includes France). Add more once you've eyeballed the real addresses.
STREET_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway",
    "apt": "apartment", "fl": "floor", "bldg": "building", "sq": "square", "ct": "court",
    "pl": "place", "ter": "terrace", "pkwy": "parkway", "rue": "rue", "sec": "sector",
    "no": "number", "num": "number",
}

_punct_re = re.compile(r"[^\w\s]")
_ws_re = re.compile(r"\s+")
_postal_re = re.compile(r"\b\d{4,6}\b")


def basic_clean(text) -> str:
    if text is None:
        return ""
    text = str(text)
    if text.strip().lower() == "nan":
        return ""
    text = transliterate_mixed_script(text)
    text = unidecode(text)
    text = text.lower()
    text = _punct_re.sub(" ", text)
    text = _ws_re.sub(" ", text).strip()
    return text


def normalize_name(name) -> str:
    """Clean + canonicalize legal-suffix tokens, but keep them (useful signal)."""
    text = basic_clean(name)
    tokens = [LEGAL_SUFFIXES.get(t, t) for t in text.split()]
    return " ".join(tokens)


def normalize_name_core(name) -> str:
    """Clean + strip legal-suffix tokens entirely. Good for blocking keys,
    since 'Acme Corp' and 'Acme Corporation Ltd' should block together."""
    text = basic_clean(name)
    tokens = [t for t in text.split() if t not in LEGAL_SUFFIXES]
    joined = " ".join(tokens)
    return joined if joined else basic_clean(name)


def normalize_address(address) -> str:
    text = basic_clean(address)
    tokens = [STREET_ABBR.get(t, t) for t in text.split()]
    return " ".join(tokens)


def extract_postal_code(address) -> str:
    text = basic_clean(address)
    matches = _postal_re.findall(text)
    return matches[-1] if matches else ""


def tokenize(text: str) -> set:
    return set(t for t in text.split() if t)


def block_key(name_core: str, width: int = 14) -> str:
    """A coarse blocking key: sorted, concatenated tokens, truncated.
    'globex international' and 'international globex' collide; typos within
    a token do not -- that's handled by the TF-IDF/embedding blocking pass,
    this key is just the cheap high-precision layer."""
    tokens = sorted(name_core.split())
    return "".join(tokens)[:width]